// Headless binding of unchanged MJPC planners to the project's RK4 Acrobot.
// The derivative adapter differentiates actual RK4 steps, not Euler dynamics.
#include <mujoco/mujoco.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "mjpc/planners/ilqg/planner.h"
#include "mjpc/planners/sampling/planner.h"
#include "mjpc/states/state.h"
#include "mjpc/tasks/acrobot/acrobot.h"
#include "mjpc/threadpool.h"

namespace {
std::atomic<std::uint64_t> step_count{0};
std::uint64_t step_limit = 10000000;
auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1200);
mjpc::Acrobot* active_task = nullptr;

void Sensor(const mjModel* model, mjData* data, int stage) {
  if (stage == mjSTAGE_ACC && active_task)
    active_task->Residual(model, data, data->sensordata);
}

void Require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}

std::vector<double> State(const mjModel* model, const mjData* data) {
  std::vector<double> state(model->nq + model->nv);
  std::copy_n(data->qpos, model->nq, state.begin());
  std::copy_n(data->qvel, model->nv, state.begin() + model->nq);
  return state;
}

void WriteTrajectory(std::ofstream& output, int control, int candidate,
                     const mjpc::Trajectory& trajectory) {
  double metadata[] = {double(control), double(candidate),
                       double(trajectory.horizon), trajectory.total_return,
                       double(trajectory.failure)};
  output.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
  for (const auto* array : {&trajectory.states, &trajectory.actions,
                           &trajectory.residual, &trajectory.times,
                           &trajectory.costs}) {
    int dimension = array == &trajectory.states ? trajectory.dim_state
        : array == &trajectory.actions ? trajectory.dim_action
        : array == &trajectory.residual ? trajectory.dim_residual : 1;
    output.write(reinterpret_cast<const char*>(array->data()),
                 trajectory.horizon * dimension * sizeof(double));
  }
  Require(bool(output), "candidate evidence write failed");
}
}  // namespace

extern "C" void reference_counted_mj_step(const mjModel* model, mjData* data) {
  auto count = step_count.fetch_add(1);
  if (count >= step_limit || std::chrono::steady_clock::now() >= deadline) {
    std::cerr << "RESOURCE_STOP count=" << count << '\n';
    std::_Exit(70);  // never throw through the native worker pool
  }
  mj_step(model, data);
}

extern "C" void reference_rk4_transitionFD(
    const mjModel* model, mjData* data, mjtNum epsilon, mjtBool centered,
    mjtNum* A, mjtNum* B, mjtNum* C, mjtNum* D) {
  // Deliberately limited to this two-hinge, no-activation-state model.
  if (model->nq != 2 || model->nv != 2 || model->nu != 1 || model->na != 0) {
    std::cerr << "DERIVATIVE_MODEL_MISMATCH\n";
    std::_Exit(71);
  }
  std::unique_ptr<mjData, decltype(&mj_deleteData)> scratch(
      mj_makeData(model), mj_deleteData);
  int sensor_count = model->nsensordata;
  auto evaluate = [&](int coordinate, double delta) {
    mj_copyData(scratch.get(), model, data);
    if (coordinate < 2) scratch->qpos[coordinate] += delta;
    else if (coordinate < 4) scratch->qvel[coordinate - 2] += delta;
    else scratch->ctrl[0] += delta;
    reference_counted_mj_step(model, scratch.get());
    std::vector<double> values(4 + sensor_count);
    std::copy_n(scratch->qpos, 2, values.begin());
    std::copy_n(scratch->qvel, 2, values.begin() + 2);
    std::copy_n(scratch->sensordata, sensor_count, values.begin() + 4);
    return values;
  };
  auto base = evaluate(4, 0);
  for (int coordinate = 0; coordinate < 5; ++coordinate) {
    bool state_coordinate = coordinate < 4;
    if (state_coordinate ? !(A || C) : !(B || D)) continue;
    auto plus = evaluate(coordinate, epsilon);
    auto minus = centered ? evaluate(coordinate, -epsilon) : base;
    double denominator = centered ? 2 * epsilon : epsilon;
    for (int row = 0; row < 4 + sensor_count; ++row) {
      double value = (plus[row] - minus[row]) / denominator;
      if (row < 4) {
        if (state_coordinate && A) A[row * 4 + coordinate] = value;
        if (!state_coordinate && B) B[row] = value;
      } else {
        if (state_coordinate && C) C[(row - 4) * 4 + coordinate] = value;
        if (!state_coordinate && D) D[row - 4] = value;
      }
    }
  }
}

int main(int argc, char** argv) {
  // mode xml original_xml context.bin output_prefix remaining_steps wall_seconds
  if (argc != 8) return 2;
  try {
    std::string mode = argv[1];
    step_limit = std::stoull(argv[6]);
    deadline = std::chrono::steady_clock::now() +
               std::chrono::seconds(std::stoi(argv[7]));
    char error[2048] = {};
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> model(
        mj_loadXML(argv[2], nullptr, error, sizeof(error)), mj_deleteModel);
    Require(bool(model), error);
    std::unique_ptr<mjModel, decltype(&mj_deleteModel)> original(
        mj_loadXML(argv[3], nullptr, error, sizeof(error)), mj_deleteModel);
    Require(bool(original), error);
    Require(mj_version() == 360 && model->opt.integrator == mjINT_RK4 &&
            model->opt.timestep == .01 && model->nq == 2 && model->nu == 1,
            "runtime/model binding mismatch");
    Require(model->actuator_gear[0] == 2 && model->actuator_ctrlrange[0] == -1 &&
            model->actuator_ctrlrange[1] == 1, "torque binding mismatch");
    for (int i = 0; i < model->nbody; ++i) {
      Require(model->body_mass[i] == original->body_mass[i], "mass changed");
      for (int j = 0; j < 3; ++j)
        Require(model->body_inertia[3*i+j] == original->body_inertia[3*i+j],
                "inertia changed");
    }
    mjpc::Acrobot task;
    task.Reset(model.get());
    active_task = &task;
    mjcb_sensor = Sensor;
    std::unique_ptr<mjData, decltype(&mj_deleteData)> data(
        mj_makeData(model.get()), mj_deleteData);
    int context_size = mj_stateSize(model.get(), mjSTATE_INTEGRATION);
    std::vector<double> context(context_size);
    std::ifstream input(argv[4], std::ios::binary);
    input.read(reinterpret_cast<char*>(context.data()), context.size()*sizeof(double));
    Require(bool(input) && input.peek() == std::char_traits<char>::eof(),
            "context size mismatch");
    mj_setState(model.get(), data.get(), context.data(), mjSTATE_INTEGRATION);
    mj_forward(model.get(), data.get());
    if (mode == "stock-rk4-prerequisite") {
      double A[16], B[4];
      mjd_transitionFD(model.get(), data.get(), 1e-6, 0, A, B, nullptr, nullptr);
      throw std::runtime_error("stock RK4 derivative unexpectedly succeeded; stop");
    }
    std::ofstream trace(std::string(argv[5]) + ".csv");
    trace << std::setprecision(17);
    if (mode == "validate-derivatives") {
      trace << "case,A_error,B_error,C_error,D_error\n";
      for (int sample = 0; sample < 16; ++sample) {
        for (int j = 0; j < 2; ++j) {
          data->qpos[j] = .8 * std::sin(1.7 * sample + .2 * j);
          data->qvel[j] = 1.3 * std::cos(1.1 * sample + .3 * j);
        }
        data->ctrl[0] = .7 * std::sin(.6 * sample);
        mj_forward(model.get(), data.get());
        double A[16], B[4];
        std::vector<double> C(4*model->nsensordata), D(model->nsensordata);
        reference_rk4_transitionFD(model.get(), data.get(), 1e-6, 0,
                                  A, B, C.data(), D.data());
        // Separate direct-step central probes never invoke the adapter.
        std::vector<double> errors(4, 0);
        for (int coordinate = 0; coordinate < 5; ++coordinate) {
          std::unique_ptr<mjData, decltype(&mj_deleteData)> plus(
              mj_makeData(model.get()), mj_deleteData);
          std::unique_ptr<mjData, decltype(&mj_deleteData)> minus(
              mj_makeData(model.get()), mj_deleteData);
          mj_copyData(plus.get(), model.get(), data.get());
          mj_copyData(minus.get(), model.get(), data.get());
          double* pp = coordinate < 2 ? &plus->qpos[coordinate] :
                       coordinate < 4 ? &plus->qvel[coordinate-2] : &plus->ctrl[0];
          double* mm = coordinate < 2 ? &minus->qpos[coordinate] :
                       coordinate < 4 ? &minus->qvel[coordinate-2] : &minus->ctrl[0];
          *pp += 1e-5; *mm -= 1e-5;
          reference_counted_mj_step(model.get(), plus.get());
          reference_counted_mj_step(model.get(), minus.get());
          auto xp = State(model.get(), plus.get());
          auto xm = State(model.get(), minus.get());
          for (int row = 0; row < 4 + model->nsensordata; ++row) {
            double reference = row < 4 ? (xp[row]-xm[row])/2e-5 :
                (plus->sensordata[row-4]-minus->sensordata[row-4])/2e-5;
            int category = (row < 4 ? 0 : 2) + (coordinate == 4 ? 1 : 0);
            double value = row < 4 ?
                (coordinate == 4 ? B[row] : A[4*row+coordinate]) :
                (coordinate == 4 ? D[row-4] : C[4*(row-4)+coordinate]);
            errors[category] = std::max(errors[category], std::abs(reference-value));
          }
        }
        trace << sample;
        for (double value : errors) {
          trace << ',' << value;
          Require(value <= 1e-4, "RK4 derivative compatibility gate failed");
        }
        trace << '\n';
      }
    } else {
      Require(mode == "sampling" || mode == "ilqg", "unknown planner");
      std::unique_ptr<mjpc::Planner> planner;
      if (mode == "sampling") planner = std::make_unique<mjpc::SamplingPlanner>();
      else planner = std::make_unique<mjpc::iLQGPlanner>();
      planner->Initialize(model.get(), task);
      planner->Allocate();
      planner->Reset(201);
      mjpc::ThreadPool pool(4);
      mjpc::State state;
      state.Allocate(model.get());
      std::ofstream candidates(std::string(argv[5]) + ".candidates.f64",
                               std::ios::binary);
      trace << "control,time,q0,q1,v0,v1,u,tip_height,native_dense_cost,mjpc_cost,steps\n";
      for (int control = 0; control <= 500; ++control) {
        mj_forward(model.get(), data.get());
        state.Set(model.get(), data.get());
        if (control < 500 && control % 2 == 0) {
          planner->SetState(state);
          planner->OptimizePolicy(201, pool);
          for (int candidate = 0; candidate < 10; ++candidate) {
            const auto& trajectory = mode == "sampling" ?
                static_cast<mjpc::SamplingPlanner*>(planner.get())->trajectory[candidate] :
                static_cast<mjpc::iLQGPlanner*>(planner.get())->trajectory[candidate];
            WriteTrajectory(candidates, control, candidate, trajectory);
          }
        }
        if (control < 500)
          planner->ActionFromPolicy(data->ctrl, state.state().data(), data->time, false);
        mj_forward(model.get(), data.get());
        int tip = mj_name2id(model.get(), mjOBJ_SITE, "tip");
        double height = data->site_xpos[3*tip+2];
        double dx = data->site_xpos[3*tip];
        double dense = (4-height)*(4-height)+dx*dx +
            .01*(data->qvel[0]*data->qvel[0]+data->qvel[1]*data->qvel[1]+
                 data->ctrl[0]*data->ctrl[0]);
        trace << control << ',' << data->time;
        for (double value : State(model.get(), data.get())) trace << ',' << value;
        trace << ',' << data->ctrl[0] << ',' << height << ',' << dense << ','
              << task.CostValue(data->sensordata) << ',' << step_count.load() << '\n';
        trace.flush();
        Require(std::isfinite(dense) && std::isfinite(data->ctrl[0]) &&
                std::abs(data->ctrl[0]) <= 1.0000000001,
                "invalid executed state/action");
        if (control == 500) break;
        reference_counted_mj_step(model.get(), data.get());
        reference_counted_mj_step(model.get(), data.get());
      }
    }
    std::cout << "complete steps=" << step_count.load()
              << " runtime=" << mj_versionString() << '\n';
    mjcb_sensor = nullptr;
    active_task = nullptr;
    return 0;
  } catch (const std::exception& error) {
    std::cerr << "INVALID_EXECUTION " << error.what()
              << " steps=" << step_count.load() << '\n';
    return 71;
  }
}
