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
#include <iterator>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
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
// Serial derivative-gate evidence only. Planning leaves this null.
std::ofstream* probe_evidence = nullptr;
int probe_case = 0, probe_variant = 0, probe_coordinate = 0;
double probe_delta = 0;

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
  std::vector<double> before;
  if (probe_evidence) {
    before.resize(mj_stateSize(model, mjSTATE_INTEGRATION));
    mj_getState(model, data, before.data(), mjSTATE_INTEGRATION);
  }
  mj_step(model, data);
  if (probe_evidence) {
    double metadata[] = {double(probe_case), double(probe_variant),
                         double(probe_coordinate), probe_delta};
    std::vector<double> after(before.size());
    mj_getState(model, data, after.data(), mjSTATE_INTEGRATION);
    for (const auto* values : {&before, &after}) {
      if (values == &before)
        probe_evidence->write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
      probe_evidence->write(reinterpret_cast<const char*>(values->data()),
                            values->size() * sizeof(double));
    }
    probe_evidence->write(reinterpret_cast<const char*>(data->sensordata),
                          model->nsensordata * sizeof(double));
    if (!*probe_evidence) std::_Exit(71);
  }
}

extern "C" void reference_rk4_transitionFD(
    const mjModel* model, mjData* data, mjtNum epsilon, mjtByte centered,
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
    if (probe_evidence) {
      probe_coordinate = coordinate;
      probe_delta = delta;
    }
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
    bool forward = true, backward = centered;
    if (!state_coordinate && model->actuator_ctrllimited[0]) {
      const double* bounds = model->actuator_ctrlrange;
      double control = data->ctrl[0];
      forward = control >= bounds[0] && control + epsilon <= bounds[1];
      backward = (centered || !forward) &&
                 control - epsilon >= bounds[0] && control <= bounds[1];
    }
    auto plus = forward ? evaluate(coordinate, epsilon) : base;
    auto minus = backward ? evaluate(coordinate, -epsilon) : base;
    double denominator = forward && backward ? 2 * epsilon : epsilon;
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
  // mode xml original_xml context.bin output_prefix steps wall_seconds [hold]
  if (argc != 8 && argc != 9 && argc != 10) return 2;
  try {
    std::string mode = argv[1];
    int control_hold = argc >= 9 ? std::stoi(argv[8]) : 2;
    Require(control_hold == 1 || control_hold == 2, "invalid control hold");
    int controls = 1000 / control_hold;
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
    Require(mj_version() == mjVERSION_HEADER && std::string(mj_versionString()) == "3.6.0" &&
            model->opt.integrator == mjINT_RK4 &&
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
    if (mode == "validate-boundaries") {
      trace << "case,centered,control,A_error,B_error,C_error,D_error,C_analytic_error,D_analytic_error\n";
      std::ofstream probes(std::string(argv[5]) + ".probes.f64", std::ios::binary);
      std::ofstream matrices(std::string(argv[5]) + ".matrices.f64", std::ios::binary);
      probe_evidence = &probes;
      Require(argc == 10, "boundary gate requires frozen probe inputs");
      std::ifstream gate_inputs(argv[9], std::ios::binary);
      for (int sample = 0; sample < 40; ++sample) {
          double values[5];
          gate_inputs.read(reinterpret_cast<char*>(values), sizeof(values));
          Require(bool(gate_inputs), "missing frozen boundary case");
          probe_case = sample;
          mj_setState(model.get(), data.get(), context.data(), mjSTATE_INTEGRATION);
          std::copy_n(values, 2, data->qpos);
          std::copy_n(values + 2, 2, data->qvel);
          double control = values[4];
          data->ctrl[0] = control;
          mj_forward(model.get(), data.get());
          int ns = model->nsensordata;
          auto direct_probe = [&](int coordinate, double delta) {
            std::unique_ptr<mjData, decltype(&mj_deleteData)> d(
                mj_makeData(model.get()), mj_deleteData);
            mj_copyData(d.get(), model.get(), data.get());
            double* value = coordinate < 2 ? &d->qpos[coordinate] :
                coordinate < 4 ? &d->qvel[coordinate-2] : &d->ctrl[0];
            *value += delta;
            probe_variant = 2;
            probe_coordinate = coordinate;
            probe_delta = delta;
            reference_counted_mj_step(model.get(), d.get());
            std::vector<double> output = State(model.get(), d.get());
            output.insert(output.end(), d->sensordata, d->sensordata + ns);
            return output;
          };
          std::vector<double> gold_A(16), gold_B(4), gold_C(4*ns), gold_D(ns);
          for (int coordinate = 0; coordinate < 5; ++coordinate) {
            std::vector<double> difference(4 + ns);
            if (coordinate < 4 || (control + 1e-5 <= 1 && control - 1e-5 >= -1)) {
              auto plus = direct_probe(coordinate, 1e-5);
              auto minus = direct_probe(coordinate, -1e-5);
              for (int row = 0; row < 4 + ns; ++row)
                difference[row] = (plus[row] - minus[row]) / 2e-5;
            } else {
              double direction = control + 2e-5 <= 1 ? 1 : -1;
              auto base = direct_probe(coordinate, 0);
              auto one = direct_probe(coordinate, direction * 1e-5);
              auto two = direct_probe(coordinate, direction * 2e-5);
              for (int row = 0; row < 4 + ns; ++row)
                difference[row] = direction * (-3*base[row] + 4*one[row] - two[row]) / 2e-5;
            }
            for (int row = 0; row < 4 + ns; ++row) {
              if (row < 4) {
                if (coordinate < 4) gold_A[4*row+coordinate] = difference[row];
                else gold_B[row] = difference[row];
              } else {
                if (coordinate < 4) gold_C[4*(row-4)+coordinate] = difference[row];
                else gold_D[row-4] = difference[row];
              }
            }
          }
          for (int centered : {0, 1}) {
            std::vector<double> A(16), B(4), C(4*ns), D(ns);
            probe_variant = centered;
            reference_rk4_transitionFD(model.get(), data.get(), 1e-6,
                                      centered, A.data(), B.data(), C.data(), D.data());
            double metadata[] = {double(probe_case), double(centered), control,
                                 data->qpos[0], data->qpos[1], data->qvel[0],
                                 data->qvel[1]};
            matrices.write(reinterpret_cast<const char*>(metadata), sizeof(metadata));
            trace << probe_case << ',' << centered << ',' << control;
            for (auto pair : {std::pair{&A, &gold_A}, std::pair{&B, &gold_B},
                              std::pair{&C, &gold_C}, std::pair{&D, &gold_D}}) {
              matrices.write(reinterpret_cast<const char*>(pair.first->data()),
                             pair.first->size()*sizeof(double));
              matrices.write(reinterpret_cast<const char*>(pair.second->data()),
                             pair.second->size()*sizeof(double));
              double maximum = 0;
              for (std::size_t i = 0; i < pair.first->size(); ++i)
                maximum = std::max(maximum, std::abs((*pair.first)[i]-(*pair.second)[i]));
              trace << ',' << maximum;
              Require(maximum <= 1e-4, "range-aware derivative gate failed");
            }
            double q0 = data->qpos[0], q01 = q0 + data->qpos[1];
            double analytic_C[] = {
                std::sin(q0)+std::sin(q01), std::sin(q01), 0, 0,
                -std::cos(q0)-std::cos(q01), -std::cos(q01), 0, 0,
                0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0};
            double analytic_D[] = {0, 0, 0, 0, 1};
            matrices.write(reinterpret_cast<const char*>(analytic_C), sizeof(analytic_C));
            matrices.write(reinterpret_cast<const char*>(analytic_D), sizeof(analytic_D));
            double c_error = 0, d_error = 0;
            for (int i = 0; i < 20; ++i)
              c_error = std::max(c_error, std::abs(C[i] - analytic_C[i]));
            for (int i = 0; i < 5; ++i)
              d_error = std::max(d_error, std::abs(D[i] - analytic_D[i]));
            trace << ',' << c_error << ',' << d_error << '\n';
            Require(c_error <= 1e-4 && d_error <= 1e-4,
                    "analytic pre-action sensor Jacobian gate failed");
          }
      }
      Require(gate_inputs.peek() == std::char_traits<char>::eof(), "extra boundary cases");
      probe_evidence = nullptr;
    } else if (mode == "validate-derivatives") {
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
      for (int control = 0; control <= controls; ++control) {
        mj_forward(model.get(), data.get());
        state.Set(model.get(), data.get());
        if (control < controls && control % (4 / control_hold) == 0) {
          planner->SetState(state);
          planner->OptimizePolicy(201, pool);
          for (int candidate = 0; candidate < 10; ++candidate) {
            const auto& trajectory = mode == "sampling" ?
                static_cast<mjpc::SamplingPlanner*>(planner.get())->trajectory[candidate] :
                static_cast<mjpc::iLQGPlanner*>(planner.get())->trajectory[candidate];
            WriteTrajectory(candidates, control, candidate, trajectory);
          }
        }
        if (control < controls)
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
        if (control == controls) break;
        for (int substep = 0; substep < control_hold; ++substep)
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
