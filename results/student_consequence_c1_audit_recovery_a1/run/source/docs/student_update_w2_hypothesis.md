# W2 hypothesis: fixed learning budget, not a selected checkpoint

**Closed after explicit user authorization on 31 October 2026.** The [W2 result](student_update_w2_results.md) reports confirmed partial retention and cost improvements without walking. The [protocol](student_update_w2_protocol.md) fixed the comparison, budget, endpoints and gates before launch. This page preserves the hypothesis motivated by the [W1 result](student_width_w1_results.md); it does not authorize other interventions or follow-ups.

W1's six validation curves worsened from the recorded 2k checkpoint to 5k and 10k while training MSE fell. Earlier checkpoints retained more controls in all six streams, but improved complete cost in only four, and none walked. This motivates testing update budget; it does not establish an optimal checkpoint or an early-stopping remedy.

Hypothesis: with the default **128-tanh** student and the same clean executed-data benchmark, a fixed 2,000-update budget improves held-trajectory reconstruction and standalone cost/control retention relative to a fixed 10,000-update budget, despite worse training reconstruction.

Use fresh paired initialization/sampler seeds, identical training-only normalization, data, features, learning targets, Adam settings and minibatch prefix. Use new paired primary/conditional confirmation resets. Fix both endpoints before fitting. The width stays 128; do not simultaneously change regularization, demonstrations, target semantics, horizon or coupling. Historical W1 checkpoints are exploratory evidence, not matched controls for a new confirmatory study.

The finite roster, budgets, useful effects, uncertainty, confirmation and stopping rules must be frozen in a new protocol before launch. The old W1 headroom is not a budget for W2. Keep reconstruction, held-trajectory transfer, cost/control retention and walking competence separate.

If 2k improves reconstruction or retention without walking, that supports the budget effect within this procedure but does not close the MPPI–policy loop. Failure would retain optimization/coverage/target and execution-sensitivity explanations. A result from this fixed contrast cannot establish a universally optimal training duration or adaptive stopping rule.
