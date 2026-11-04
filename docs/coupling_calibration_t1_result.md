# T1: calibrate the mean-prefix reconstruction scale

**The frozen choice is α=31.734901633936317. It changes some weighted actions, but leaves every top candidate unchanged. This is saved-bank calibration, not executed teacher or learning evidence.**

We used six banks (K256/H48) from three student states at controls 0, 50 and100 of the **same first primary evaluation**, with two same-state refinements each. All bind the final L3 s0 coupled actor. They are six banks, not six independent states; p1 starts from p0's refined nominal without executing a control. Temperature stays3.

For each valid candidate, R=D/N is half-summed torque error divided by its actual executable-prefix length. For each basis, α=q/median_bank[std(R)/std(C)] (population std; equal bank weighting). The competitive basis is the **32 lowest finite task costs**, selected without R, then treated uniformly for spread calibration:

| Target q | α, all finite candidates | α, competitive32 |
|---|---:|---:|
| 0.1 | 9.2357231425214206 | 10.57830054464544 |
| 0.3 | 27.707169427564256 | 31.734901633936317 |
| 1.0 | 92.357231425214195 | 105.7830054464544 |

The protocol selected competitive q=.3 **before** execution or fitting. It gives a median achieved spread ratio .3, but individual banks range .219–.521. This data-dependent scale heuristic does not guarantee a strong intervention: competitive32 contains at least99.99963% of baseline softmin mass while ESS is near1. Unweighted spread over32 candidates differs from variation among candidates carrying appreciable weight.

At the selected α, the counterfactual score is C+αR:

| State/pass | Top index | TV | ESS, α0→selected | Fallen mass, α0→selected | Censored / last-step fall mass | Nonfallen remainder | First-action RMS |
|---|---|---:|---|---|---|---:|---:|
| 0/p0 | 63→63 | 0.002858 | 1.0109→1.0051 | 1.34e-06→6.91e-07 | 6.91e-07 / 2.52e-15 | 0.999999 | 0.0016151 |
| 0/p1 | 148→148 | 0.03266 | 1.1038→1.0328 | 8.46e-12→5.18e-12 | 5.18e-12 / 6.88e-24 | 1.000000 | 0.010201 |
| 50/p0 | 105→105 | 7.384e-07 | 1.0000→1.0000 | 1.23e-07→8.45e-07 | 8.45e-07 / 0 | 0.999999 | 2.9775e-07 |
| 50/p1 | 103→103 | 0.04187 | 1.1296→1.0390 | 5.57e-07→1.41e-07 | 7.78e-09 / 1.34e-07 | 1.000000 | 0.019937 |
| 100/p0 | 238→238 | 0.01146 | 1.0101→1.0278 | 0.00484→0.0136 | 0.0136 / 2.01e-06 | 0.986418 | 0.0018828 |
| 100/p1 | 59→59 | 0.085 | 1.0139→1.2061 | 3.03e-07→7.61e-05 | 7.61e-05 / 0 | 0.999924 | 0.021063 |

Top changes: **0/6**. First-action RMS exceeds .01 in3/6 banks, all second refinements; this is not the future executed-controller criterion. Fallen weight increases in3/6 banks, including .00484→.01358 at state100/p0. No teacher-competence or learning benefit follows from these reweightings, and none of these proxies admits or rejects demonstrations.

Zero-length candidates: 0; finite zero-R candidates: 0. Last-control falls: 10 candidates across banks. A fall at N=48 contributes fallen mass but is **not censored** (N<48); these are reported separately. In `measurements.npz`, the historical key `b*_D` stores normalized R, not the original summed D.

Proceed with exactly **31.734901633936317** for the already-declared T0 comparison because it is the preregistered competitive-.3 choice, not because it produced the largest observed change. T0 must determine actual action intervention and executed survival/task outcomes. No T0 was run for this report. [Exact outputs, all36 rows, source bindings and costs](../results/coupling_calibration_t1_t0_001/t1/README.md).
