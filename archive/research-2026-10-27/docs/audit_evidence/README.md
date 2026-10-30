# Audit evidence index

Audit date: 23 October 2026. Baseline: `01eb594ccba3780f29179f14db3caf9755c5d6dc`.

- [Literature report](../related_work.md): candidate claims, budget axes, search protocol, prior-art matrix, threat note and required baselines.
- [Code and validity report](../code_audit.md): severity-ranked findings, fixes, test adequacy, reproductions and remaining gates.
- [Theory review](theory_review.md): individual assertions checked against primary material and executable counterexamples. The original `docs/theory.md` is preserved.
- [Core audit](core_audit.md): independent label calculation, reference MPPI comparison, CPU/GPU checks, profiles, coverage and mutation results.
- [Evaluation audit](evaluation_audit.md): observable reset-use decisions, runner seed domains, Walker task differential, holdout reservation and power assumptions.
- [Mordatch 2015 crosscheck](mordatch2015_review.md): paper and official reviews/rebuttal, extending the 2012/2014 CIO comparison.
- [Audit manifest](audit_manifest.json): source hashes, checkout locations and execution status.

Regression files ending in `_red.txt` preserve failing tests before their corresponding fixes; `_green.txt` files preserve passing checks. Full-suite, static-analysis and mutation outputs include adverse findings rather than only successes. The reproduction JSON files distinguish exact controlled replay, cross-machine checkpoint evaluation, and official-data statistic reanalysis. None is mislabeled as fresh VAMPS training.

The live server receives documentation only at `robopil5090:/home/jaisel/mppi-gps/docs`. Source fixes and new tests are in the isolated `/home/jaisel/mppi-gps-audit-20261023` checkout. Existing training processes continue against their original source.

Primary papers and tracked files from the three pinned official comparator repositories are archived outside the project checkout at `robopil5090:/home/jaisel/mppi-gps-audit-materials-20261023/primary-materials.tar.gz`; see the [archive manifest](literature/material_archive.json). This preserves the inspected inputs without adding large upstream repositories to the project's git tree. It does not broaden the reading scope recorded in the [source ledger](literature/source_read_ledger.json). The two large raw GPU traces are preserved in the same server materials directory under `gpu_original_trace.json` and `gpu_fixed_trace.json`; the smaller tables and numerical arrays are included here.

The full Mordatch2015 paper, extracted text and rendered pages are preserved
locally under `runs/2026-10-24_literature_materials_preserved_001/mordatch2015/`,
outside the published evidence. Its [read manifest](mordatch2015/read_manifest.json)
retains the upstream URL, content hash and reading scope; our review is linked above.

Final reset range **5,000,000–5,000,199 was reserved and not evaluated**. The conditional power calculation is a planning result, not authorization to launch 800 training runs or evidence that the present method meets its performance claim.
