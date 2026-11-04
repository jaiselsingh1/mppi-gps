# T1 saved-bank calibration

[One-page interpretation](../../../docs/coupling_calibration_t1_result.md).
Frozen competitive-.3 choice: **31.734901633936317**. No executed-controller,
standalone walking or learning outcome is measured here.

- [Exact frozen config](frozen_config.json), [result](result.json), [all36 rows](per_bank.csv), [saved weights and measurements](measurements.npz)
- [Selected coefficient, six banks](selected_counterfactual.csv), [zeros/censoring/last-step falls](candidate_categories.csv)
- [Input inventory and source episode/control bindings](input_inventory.json), [input preparation cost](input_preparation_receipt.json)
- [Frozen calibration implementation](source/coupling_calibration.py), [frozen bank update](source/bank_review.py)

All copied outputs preserve their original bytes. Large full candidate banks,
H5 records and checkpoints stay at the explicit local paths and SHA256 bindings
inside the config/inventory. Six banks share three physical states from the first
primary evaluation of one final actor. This publication performs no controller
execution, policy forward, fitting or T0 rollout.

`measurements.npz` uses `b*_D` for the normalized mean-prefix residual R=D/N.
Top32 membership is determined by task cost alone; calibration uses an unweighted
population standard deviation within that subset. Near-unit ESS means an imposed
spread ratio does not ensure a large change to weighted actions. Candidate
proxies are not teacher-competence, demonstration-admission or learning gates.
