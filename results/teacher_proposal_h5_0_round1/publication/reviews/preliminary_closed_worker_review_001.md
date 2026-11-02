# Preliminary closed-worker review — independent scientific audit pending

These are recomputed scalar declarations, not an independent raw-evidence audit pass.

| Arm | Walking /12 | Full survival /12 | Controls | Mean complete cost |
|---|---:|---:|---:|---:|
| independent | 12 | 12 | 12,000 | 364.579995 |
| selector_only | 11 | 11 | 11,201 | 695.369860 |
| proposal_guided | 11 | 11 | 11,199 | 692.136979 |

| Stream | Guided arm | Mean cost / paired I mean | Cost change | Walking gate | 5% cost gate |
|---|---|---:|---:|---|---|
| 0 | selector_only | 1.044631669 | +4.463167% | True | True |
| 1 | selector_only | 3.561842263 | +256.184226% | False | False |
| 2 | selector_only | 1.048469043 | +4.846904% | True | True |
| 0 | proposal_guided | 1.003901561 | +0.390156% | True | True |
| 1 | proposal_guided | 1.007677398 | +0.767740% | True | True |
| 2 | proposal_guided | 3.719889328 | +271.988933% | False | False |

Reported work: 852,903,936 model controls + 34,400 live controls = 852,938,336 native controls; 3,411,753,344 physics substeps. NN work is separately 1,087,519,492 forward rows, 1,536,000 backward rows and 6,000 Adam updates, all learning from the three parent warm-ups.

Expected independent prediction audit: 1,085,983,492 rows in 103,308 forward calls and six checkpoint loads, with no training or native replay. This is an expected audit census, not completed audit work.

Child fits, standalone walking/survival/cost, child reconstruction/held error and paired teacher–student gaps were not measured. No second round or automatic follow-up is permitted.
