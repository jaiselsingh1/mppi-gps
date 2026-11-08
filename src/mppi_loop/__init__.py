"""MPPI loop: MPPI explores, a similarity on where its rollouts go steers it, the policy learns.

Modules, in reading order:
- loop.py        the rounds: MPPI walks -> similarity reference -> policy fit -> swap-in -> judge
- similarity.py  the state-set similarity added to MPPI's rollout cost, and the medoid walk
- fit.py         the policy fit: mixed targets, LQR-gain slopes, the policy-MPPI gap
- dynamics.py    batched stepping, linearization, LQR gains, per-control task costs
- policy.py      the policy network
- spec.py        what the policy observes and what counts as doing the task (Walker)
- measures.py    stride section and walk spread (logging only)
- tasks/         walker.py, hopper.py: starts, starting walks, starting policy, judge, task score
"""
