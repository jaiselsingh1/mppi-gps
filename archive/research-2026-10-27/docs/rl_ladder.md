# S2: local Walker policy ladder

Starting commit: `86a29d7bafc5af8e3c22235a1e7269b02535f078` on `main`.
Question: can task-matched SAC produce policies of different measured ability
without changing the frozen MPPI teacher? Implement and validate the adapter,
then the learner, then run a plumbing smoke test before a full training ladder.
No gate, teacher tuning, value function or distillation is part of S2.

## Fixed task and learning choices

- Physics and task: local `Walker2d`, `assets/walker2d.xml`, frozen
  `configs/walker2d_task.json`; 0.002 s physics, four substeps per action.
- Reward: negative executed task cost. Only on a fall, additionally charge
  `-gamma * unhealthy_cost_weight / (1 - gamma)`, or -995 at gamma=0.995.
  The live cost already includes the first unhealthy step and terminal penalty.
  The absorbing continuation has no actions or entropy bonus.
- A fall terminates; the 1,000-control limit truncates. Save the actual next
  observation before resetting; only termination masks the SAC bootstrap.
  A fall on control 1,000 is still a termination.
- SAC reuses the archived twin-Q, squashed-Gaussian learner's architecture and
  update equations, not its Gymnasium environment, reward, checkpoint or GPS
  code. Actor/critic hidden layers: 256,256; actor LR 0.0003, critic/alpha LR
  0.001; batch 256; tau 0.005; policy frequency 2, target frequency 1; alpha
  starts at 0.2 with entropy tuning; 5,000 random warm-up actions.
- No new running observation normalizer: the original local 17-value observation
  and velocity clipping are retained. Identity mean/std buffers are embedded in
  checkpoints so training, standalone evaluation and later gate inference use
  exactly the same transform. No reward scaling.
- CPU with one Torch thread and one simulator rollout thread. No remote jobs,
  external experiment tracker or GPU/backend change.

These are starting settings, not claimed optimal. SAC is discounted and entropy
regularized; it shares task costs and failure semantics with MPPI, not precisely
the same objective. References: [SAC equations](https://spinningup.openai.com/en/latest/algorithms/sac.html)
and [termination versus truncation](https://gymnasium.farama.org/tutorials/gymnasium_basics/handling_time_limits/).

## Protocol fixed before training

One training seed (0) initially; this first ladder is developmental, not evidence
across independent training seeds. Save the untrained policy and every 25,000
steps through 1,000,000. Checkpoints are policy evaluation snapshots, not exact
training restart points unless replay, simulator and RNG state are also saved.
No old checkpoint is restored. A separate short plumbing smoke test must finish
with finite losses and a loadable checkpoint before starting this run.

Commands (use a fresh run directory every time):

```sh
.venv/bin/python -m scripts.training.train_walker2d_sac \
  --run-dir runs/NEW_s2_ladder_seed0 --classify-at-end

.venv/bin/python -m scripts.runners.eval_policy_walker2d \
  --checkpoint runs/NEW_s2_ladder_seed0/checkpoints/step_0025000.pt \
  --output runs/NEW_policy_evaluation/metrics.json \
  --video runs/NEW_policy_evaluation/walker.mp4
```

Both entry points use tyro. Training writes `status.json`, `config.json`,
`episodes.jsonl`, update/monitoring `metrics.jsonl`, hashed `checkpoints.jsonl`,
and source snapshots. `--classify-at-end` evaluates every saved checkpoint on
the 30 classification seeds after training and writes `classification/ladder.json`
with the earliest member of each measured category. Missing categories are
explicit. Standalone and monitoring evaluations retain raw NPZ traces alongside
their reports. The standalone command renders only the first episode, using
the same tracking camera and video timing as the MPPI runner.

Standalone evaluations are deterministic, 1,000 steps, same reset seeds per
checkpoint. Use reset seeds 50,000–50,029 for classification; do not call them
untouched test data. Monitoring may use 60,000–60,004. Training resets come from
its own seeded RNG. Later held-out comparison seeds must be chosen disjointly
from all observed seeds and recorded before S4.

Development labels, deliberately not a formal qualification:

- **weak:** fewer than 15/30 full episodes;
- **strong:** at least 29/30 full episodes AND mean episode speed MAE <= 0.20 m/s;
- **intermediate:** all other checkpoints.

Classify from measurements, not training step or return. All failed episodes
remain in completion and episode-averaged speed MAE (over executed steps, as in
the frozen runner); this conditional speed metric is never used without the
completion result. Select the earliest checkpoint meeting each category, and
report missing categories honestly. No selection on final S4 test seeds.
H2's preselected strong-checkpoint metric is speed MAE, with completion also
required to meet the pre-set non-inferiority criterion. Selection is not a claim
that the policy is stronger than MPPI.

## S3/S4 boundary

Freeze and classify the saved policies before gate experiments. Shadow action
acceptance rate does **not** determine S4 sample size. Use a separate active-gate
development pilot of full paired episodes, or conservative joint-outcome
assumptions; assess power for both completion and speed MAE passing together.
Freeze final sample counts before untouched evaluation. Preserve the power
script, assumptions, seeds and outputs. The all-success 189-pair boundary is
not a general power guarantee. Margins remain 2 percentage points and 0.02 m/s.
