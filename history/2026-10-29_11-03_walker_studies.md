# 2026-10-29 to 11-03: preregistered Walker studies (old main, Codex)

**Code and full write-ups:** tag `archive/main-20261107`. Each study has
`docs/<study>_protocol.md` and `docs/<study>_results.md`, and most have an evidence capsule in
`results/<study>/`. Start with `docs/research_tracks.md` and `docs/results.md`. Raw runs are in
the main checkout's `runs/` and `artifacts/` (not in git); `docs/evidence_availability.md` maps
each external file, some kept only on the remote host.

The student in these studies is a 17 → 128 tanh → 128 tanh → 6 torque network. "Walks" means the
full 8 s episode at the target speed; panels are 16 or 32 fresh standing starts.

| Study | Question | Result |
|---|---|---|
| CIO Walker v1 | Does a CIO-style loop transfer? | MPPI 4/4; policy 0/16; no feedback round ran |
| v2 | Same, with fitting and recovery fixes | Policy 0/16 (mean 231 of 1,000 controls). Higher-weight guided MPPI walked 2/2 at +1.8% cost but cut disagreement only 0.07% (5% required), so its data were rejected. Six fitting variants: all 0/16. A corrective-data child: 0/16 |
| Loop v3.1 | Does guided MPPI data train a better child than independent data? | Small cost advantage, worse survival; not closed |
| L1 | Can this student learn walking from successful teacher data (clipped vs full velocities)? | Both 0/16 |
| H4.1 | Does halving the agreement weight after a child fails help the next round? | All 6 teachers walked; all 3 children got worse than their parent |
| K1 | Does agreeing with MPPI's actions mean competence? | Clones agree with MPPI better than SAC does, yet fail; SAC walks 16/16 + 16/16 |
| W1 | Wider student (128 → 256)? | Training error −48%, held-out error +24%, all 0/16 |
| W2 | Stop at 2k instead of 10k updates? | 2k retains more and costs less; still 0/16 |
| D1/D2 | Transfer diagnosis by replay and single-action swaps | Support does not certify competence |
| S1, S1.1 | Does the data source matter (MPPI vs SAC reference)? | S1 interrupted by a host reboot. S1.1: reference 32/32; MPPI-data children 0/96; reference-data children 5/96 (one seed) |
| K1.1, K1.2 | How do trajectories and motions differ? | Close overall scores hide opposite local differences; no transfer cause found |
| H5.0 r1 | Do proposal/selector guidance keep the teacher competent? | Independent 12/12, guided 11/12 each; stopped before any child fit |
| C1.0 | Weight training rows by their effect on cost? | Every parent and child failed |
| S1.2 V | Can the SAC reference recover from the student's own states? | Walks from 27/72, survives 46/72, vs 0/72 for the parents |
| J1 | Does a similarity score predict which data teaches? | Not enough outcome variation to test; closed |
| SAC continuation (11-02) | Can SAC carry on from MPPI's states? | Kept walking from 534/560 MPPI states (640/640 of its own); one MPPI action was no better than a random one. Cloning SAC's own 9,000 rows walked 94/96 |
| J2 | Does label jitter cause cloning failure? | Clean-data fits walked 5, 15, 32, 30, 30, 31 of 32 (143/192), below the 173 gate, so the jitter arm never ran. Post hoc: error on the student's own states tracks walking; error on reference states does not |

**Overall:** no MPPI-taught student walked in this period; SAC's data cloned easily.
