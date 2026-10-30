# Independent expert value and policy withdrawal: implementation validation

The recovered formulation is implemented and both bounded server validations completed. This validates execution of the mechanism, **not learned Walker locomotion**: the full eight-second validation policy completed 0/16 episodes, with mean duration 0.5185 s and velocity MAE 1.724435 m/s. No large training or sample-efficiency benchmark was run.

## Exact formulation

The policy controls its entire first L decisions, then an independent MPPI oracle supplies the remaining finite-horizon cost. Policy parameters are searched through antithetic derivative-free perturbations using

`J_L(theta) = E[sum(t=0..L−1) gamma^t * cost(s_t, pi_theta(s_t)) + gamma^L * V_MPPI(s_L, N−L)]`.

The expert value is supervised only on actual independent-MPPI continuation costs, conditioned on 17 observation features and remaining/N. It is **V_MPPI, not V_pi**, an accepted-plan Bellman target or a hybrid default/policy value. Teacher actions are never cloning targets. Each expert query starts with its own zero plan, then retains its own receding-horizon plan; learner actions, policy tails, tracking penalties and learned values do not enter MPPI.

The expert value is frozen during each parameter search. A proposed update is accepted only when actual takeover costs improve strictly on both training and separate verification starts, paired by planner seeds. Predicted improvement alone cannot update the actor. Prefix length grows under the declared curriculum. At **L=N**, remaining cost is exactly zero and scoring invokes neither MPPI nor the expert-value network. Falls receive one terminal charge plus the finite discounted absorbing suffix; no unobserved infinite tail is reused. See [the implementation description](../../expert_withdrawal.md).

## Controlled counterexample

The two-action cost is `0.25*a0^2 + (a0−1)^2 + (a1−a0)^2`. The weak `(0,0)` policy costs **1**. Taking the expert's first action and reverting to the weak policy, `(1,0)`, costs **1.25**: a negative reward advantage of −0.25 under that weak continuation. The complete expert `(1,1)` costs **0.25**. Outcome-based prefix search followed by full withdrawal learns approximately **(0.8,0.8), cost 0.20**, with different actions from the expert and no action targets. This is a controlled mechanism regression, not a Walker result.

## Validation runs

Frozen source: `66353f1d2c3f6d2a0cb71ee05623a6a43b0007a4`. The full regression suite passed **870 tests**, followed by **two additional toy checks**. Both real-server runs report source unchanged, retain source copies and checkpoints, and passed the accepted-update checks. For every accepted update, measured proposed cost was strictly lower on both panels.

| Check | Mixed-prefix smoke | Full-withdrawal smoke |
| --- | --- | --- |
| Finite horizon N | 64 | 1,000 |
| Executed prefix lengths | 1, 64, 64 | 1,000, 1,000, 1,000 |
| Completed parameter updates / accepted | 3 / 1 | 3 / 2 |
| Expert-value fits | 1 | 0 |
| Expert assistance at each update | Yes, no, no | No, no, no |
| Actual teacher controls / planner calls | 1,010 / 1,010 | 0 / 0 |
| Teacher candidate-planning controls | 32,320 | 0 |
| Policy controls | 1,963 | 9,492 |
| Standalone evaluation controls, initial + final | 512 | 2,071 |
| Final standalone evaluation | 4/4 complete 64 controls | 0/16 complete 1,000 controls |
| Final mean duration / velocity MAE | 0.512 s / 1.712283 m/s | 0.5185 s / 1.724435 m/s |
| Walking qualification | Not applicable to 64-control smoke | Failed |

Mixed-prefix overrides used two perturbation pairs, two training/calibration/verification starts, 50 value updates and K=8/H=4 independent MPPI with one initial pass. Full withdrawal used four perturbation pairs, four training and verification starts, and three full-N updates; configured K=32/H=48 was unused. Both used seed 0, gamma 0.999, two process workers and maximum parameter-step norm 0.05. Curriculum completion here means the configured stage sequence executed; it does not mean walking was learned.

All 18 archived source files per run and all retained checkpoint fingerprints were checked against their manifests. The full-N ledger has zero teacher execution, teacher planning and expert-value fits; every update records expert assistance false. Short mixed returns remain takeover-controller outcomes, not standalone returns. The full-N result is the actual independent policy check and remains poor.

## Preserved local proof

Generated JSON, NPZ, test reports and checkpoints remain local under the [artifact policy](../../artifact_layout.md). Root directory: `runs/expert_withdrawal_validation_20261025_001/`. Links: [validation manifest](../../../runs/expert_withdrawal_validation_20261025_001/validation.json), [full regression XML](../../../runs/expert_withdrawal_validation_20261025_001/pytest.xml), [mixed protocol](../../../runs/expert_withdrawal_validation_20261025_001/mixed_prefix_001/protocol.json), [mixed update decisions](../../../runs/expert_withdrawal_validation_20261025_001/mixed_prefix_001/metrics.jsonl), [full-N protocol](../../../runs/expert_withdrawal_validation_20261025_001/full_withdrawal_001/protocol.json), [full-N update decisions](../../../runs/expert_withdrawal_validation_20261025_001/full_withdrawal_001/metrics.jsonl), and [full-N final evaluation](../../../runs/expert_withdrawal_validation_20261025_001/full_withdrawal_001/final.json).

| Local artifact | SHA-256 |
| --- | --- |
| `validation.json` | `681e56153cc1cdc850a6ed046032e78c29f0c58703f8af6f9db6a32746c6c7f8` |
| `mixed_prefix_001/protocol.json` | `63ac034d620d92d0334f3c1cbdc815a40fc7e3606f4ceb8a70c120d361275271` |
| `mixed_prefix_001/policy_final.pt` | `34426e925dee776a5798a9b6225a027944b2a14ea65a655175cedca97cb2b3c7` |
| `mixed_prefix_001/final.json` | `258058b99f00463e7e133d1dca9205ac0843fa816ef2aff83ec9be7fb1c6c79d` |
| `full_withdrawal_001/protocol.json` | `cc82117df7172fc353a969c9f8c59fcfc097074cfd4554dad38aa3de1c0168ee` |
| `full_withdrawal_001/policy_final.pt` | `6bdb7fda5a981e7331873e47c59d05aa158be2b39b3b8cd8ab805ab2f675ca9d` |
| `full_withdrawal_001/final.json` | `9f81de4a975a1d28c5bb014e93db8f2202996a7a38a121c580d5d0f3c852c869` |

This implementation supplies a bounded alternative to action cloning. It does not establish novelty, reliable extrapolation of the expert value, curriculum success beyond these smokes, stable policy search at scale, cross-seed robustness, or an exploration/sample-efficiency advantage. Sampled acceptance is not a performance guarantee outside its panels. The completed earlier studies remain unchanged.
