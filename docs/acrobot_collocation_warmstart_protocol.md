# Acrobot saved-prefix collocation continuation, v2.0

This tests initialization after nine preserved Hermite–Simpson attempts from
smooth guesses failed. Their failure did not establish infeasibility. The new
question is whether an already executed MPPI prefix can initialize independent
CasADi/IPOPT exact-RK4 trajectory optimization on the identical physical model.
There is no new MPPI execution, policy fitting, gain change, controller change,
cost tuning, model change, or loosened tolerance in this continuation.

The complete configuration is `configs/studies/acrobot_collocation_warmstart_v2_0.json`.
The unchanged XML SHA256 is
`324c8a6bf3bfea0b7faf72395dd3ac7a75cc9c2047159d1f1cd212aeee203cbe`:
MuJoCo 3.6.0, RK4 .01 s, damping .05, normalized input [-1,1], physical
elbow torque [-2,2] Nm. The existing 32 native one-step and 32 Drake nonlinear
equation checks are bound by their saved SHA256 receipts; no parity physics is
repeated. The symbolic equations and optimizer are reused from
`src/gps/mppi_gps/references/acrobot_collocation.py`.

Preparation reads only the saved `swing-development/evidence.h5` episodes
`trials/variant_0/trial_0` and `trial_1`, and the saved recovered LQR matrix P.
For each episode separately, wrap each saved state's two angles to [-pi,pi),
calculate x'Px over all 601 saved states, and select the earliest minimum.
No optimization result affects this selection. Expected prefix lengths are
569 and 508 controls at 50 Hz: 11.38 and 10.16 s. Preserve full frozen start
contexts from `acrobot_attractor_v1_0.json` and verify exact equality to the
recorded state, warmstart and integration arrays. Export selected raw prefixes,
all 601 selection scores, and their provenance before any optimizer/native run.

Each saved action becomes physical torque twice its value and is held for two
100 Hz steps. The fine initial guess interleaves each preserved native 50 Hz
state with one midpoint from the existing symbolic .01 s RK4 step. Preserve all
native endpoints, including the selected endpoint. Do not replace that endpoint
with the desired terminal state: the initial guess may violate the terminal
constraint while retaining small dynamics defects. The terminal equality is the
independent nearest 2pi winding of each selected endpoint angle and exactly
zero velocities. Solve once per prefix at its fixed duration, using the existing
fine minimum-effort objective, 1000 iterations, 180 solver CPU seconds and
1e-8 feasibility tolerance. IPOPT bound relaxation is zero; candidate controls
must lie strictly within [-2,2] before any native replay. Save initial guess,
initial constraint diagnostics, failed/debug arrays, solver statistics and logs.

Commit source, configuration and protocol before launch. Preparation freezes
hashes of every source/model/include/config/input/parity file and the exported
prefix arrays. All commands verify that manifest before doing work. Export a
create-only source tree and preparation directory beneath the configured new
remote root; do not overwrite any previous attempt. The existing isolated
CasADi 3.7.2 environment supplies the solver. The existing native MuJoCo Python
environment supplies replay. No dependency installation is required.

Run at most two solver attempts. An ordinary infeasible solve advances to the
other frozen prefix; invalid input, nonfinite data, failed binding or exhausted
budget stops the route. After the two attempts, replay each feasible candidate
at most once, in roster order, restoring its saved full native initial context,
with exactly 200 zero-torque native steps appended. No clipping or feedback is
permitted. The final 200 post-step states must all have height >=3.8 m,
wrapped angle magnitudes <=.02 rad and speed magnitudes <=.05 rad/s. Report
native-versus-optimizer maximum absolute state error before the suffix, final
state, terminal and hold errors, task cost, official dense reward, torque effort,
wall/CPU time, output size and all native-step counts. Preserve partial replay
arrays on invalid physics and stop; do not retry. Maximum possible work for
both prefixes is 1138+200+1016+200=2554 native steps, below the hard 3000 cap.

Hard limits for the entire continuation are 360 solver CPU seconds, 600 combined
wall seconds, 64 MiB new output, two solves and 3000 native steps. Source export
and remote process supervision must include logs and outputs in these limits;
all stages use one math thread. No follow-up, tolerance adjustment, horizon
extension, extra start, or new reference method follows this route. The result
is either an identical-model native-executing feasible reference or a retained
failure to produce one under these bounds. A successful trajectory supplies a
feasible upper bound at its fixed duration; it establishes no minimum time,
global optimum or nonlinear funnel. The previously derived 1.962 s nominal
continuous energy-time lower bound remains separately identified, and neither
the saved development starts nor their native reward reproduce the official
standard-start benchmark.

Commands, from the frozen source root, are:

```text
python -m src.gps.mppi_gps.references.acrobot_warmstart prepare --config configs/studies/acrobot_collocation_warmstart_v2_0.json --out <run>/preparation
<casadi-python> -m src.gps.mppi_gps.references.acrobot_warmstart solve --config configs/studies/acrobot_collocation_warmstart_v2_0.json --preparation <run>/preparation --authorization <run>/launch.json --out <run>/solves
<native-python> -m src.gps.mppi_gps.references.acrobot_warmstart native --config configs/studies/acrobot_collocation_warmstart_v2_0.json --preparation <run>/preparation --solve-dir <run>/solves --authorization <run>/launch.json --out <run>/native
```

Preparation performs saved-array arithmetic only. Solve/native stages require a
commit-bound launch authorization receipt supplied by the coordinating agent;
preparation alone does not authorize execution. Runtime supervision is part of
the launch receipt and must enforce the remaining wall/output limits even if a
solver or native process exits unexpectedly.
