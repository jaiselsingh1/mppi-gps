# T1: calibrate the mean-prefix reconstruction scale

**Use α=31.734901633936317 for the declared T0 comparison because the protocol fixed competitive target .3. These saved-bank counterfactuals establish neither teacher competence nor learning benefit.**

Six H48/K256 banks bind the final L3 s0 coupled actor: two same-state refinements at controls 0, 50 and 100 of the **same first primary evaluation**. They represent three physical states. The second refinement uses the first refined nominal without executing a control.

The score is C+αR, with R=D/N (half-summed torque error per actual executable control). For target q, α=q/median_bank[std(R)/std(C)], using unweighted population spreads and equal bank weighting. The competitive subset is the 32 lowest finite task costs, selected independently of R or later outcomes. Temperature remains 3.

Each row below aggregates the six fixed banks; fall/censoring columns are mean weight masses, not episode failure rates:

| Basis / target q | α | Top changes | Median TV | Mean fallen / censored mass | First-action RMS range |
|---|---:|---:|---:|---|---|
| all/0.1 | 9.2357231425214206 | 0/6 | 0.004704 | 0.0007375 / 0.0007372 | 3.24e-08–0.0082 |
| all/0.3 | 27.707169427564256 | 0/6 | 0.01974 | 0.00178 / 0.00178 | 2.14e-07–0.0185 |
| all/1 | 92.357231425214195 | 1/6 | 0.05051 | 0.1922 / 0.1922 | 2.64e-05–0.181 |
| competitive/0.1 | 10.57830054464544 | 0/6 | 0.005605 | 0.0007623 / 0.000762 | 3.88e-08–0.00919 |
| competitive/0.3 | 31.734901633936317 | 0/6 | 0.02206 | 0.002277 / 0.002276 | 2.98e-07–0.0211 |
| competitive/1 | 105.7830054464544 | 2/6 | 0.0513 | 0.2854 / 0.2854 | 7.93e-05–0.258 |

The selected competitive-.3 coefficient leaves all six top candidates unchanged. Three banks exceed .01 first-action RMS, all second refinements; the range is approximately 3e-7–.0211. Fallen weight increases in three banks, including .00484→.01358 at state100/p0. [Per-bank detail](../results/coupling_calibration_t1_t0_001/t1/selected_counterfactual.csv) preserves every selected-condition change and remaining nonfallen mass.

Target .3 is a **data-dependent scale heuristic**, not a guarantee of strong action change. Individual achieved spread ratios range .219–.521. The competitive32 subset carries at least 99.99963% of baseline softmin mass, yet calibration treats those candidates uniformly while baseline ESS is near 1. Variation across that subset therefore differs from variation among candidates carrying appreciable weight.

Zero-length and finite zero-R candidates both number zero. Ten candidates fall on the last control: these contribute fallen mass but are not censored (N<48). Those categories remain separate. In the saved measurement archive, `b*_D` stores normalized R rather than original summed D.

The frozen competitive-.3 rule—not the largest apparent effect—determines the coefficient. T0 must measure actual executed action changes, survival and task outcomes; these proxies neither admit demonstrations nor establish teachability. [Exact outputs, all 36 rows, versioned aggregate table, source bindings and costs](../results/coupling_calibration_t1_t0_001/t1/README.md). No T0 execution or inference was performed for this report.
