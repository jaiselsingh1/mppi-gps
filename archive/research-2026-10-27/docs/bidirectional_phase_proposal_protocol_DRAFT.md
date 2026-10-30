# DRAFT: bounded native phase proposals for Acrobot discovery

**Not implemented, frozen, launched or qualified.** This document was prepared
without reading the active temporal-assay or Ending-v3 results. It changes no
existing source, run, task, criterion or budget. Its only immediate product is
a reviewable conditional proposal. The original scientific negatives remain.

## Question and activation condition

If the closed, independently audited block-versus-knot assay still fails to
discover a complete capture, inspect its saved banks before considering this
draft. This proposal addresses one specific possible gap: independent temporal
proposals are anchored to each window and the retained nominal has a constant
tail, with no explicit representation of oscillation phase across windows.
It does not assume that phase representation is the observed cause, that a
short horizon can score the useful investment, or that larger energy implies
capture. The prior [static diagnosis](/Users/jaisel/Documents/mppi-gps/docs/bidirectional_acrobot_discovery_diagnosis.md)
records the source and evidence supporting this narrower hypothesis.

Two gates remain distinct. A small proposal-coverage assay can test native
waveform coverage. Establishing the intended local-competence/ending-failure
contrast, or moving on to ordinary Acrobot actor learning, additionally needs
a **strong independent long-lookahead executed feasibility witness** on the
unchanged native task, bounds, reset distribution and final slow-capture
criterion. The failed direct energy/LQR heuristic is not that witness.
A long hypothetical plan, a chosen seed, a different torque limit, or old
position-only hold scores do not suffice. The witness must have its own frozen
source, complete actual closed-loop trajectories, fresh confirmation and paid
query ledger. Its costs cannot be hidden in this short-horizon comparison.
No witness source, extra budget or automatic launch is authorized by this draft.

## One concrete candidate representation

Keep K512 Gaussian slots,16 memory/fallback slots and **exactly32 broad slots**.
The proposed treatment replaces those32 broad slots, rather than adding more
rows. It uses native torque waves with two waveform types, two fixed periods
and eight phases:

```
P in {.9134159022626218, 2.356877965498118} seconds
phi in {2*pi*j/8 : j=0,...,7}
wave in {sin(theta), square(theta)}
theta(t) = 2*pi*t/P + phi
square(theta) = +1 if sin(theta)>=0, otherwise -1
u[k] = wave(theta((elapsed+k)*.01)), k=0,...,h-1
```

The periods come from a source-bound undamped small-angle calculation at
hanging rest. They are a finite disclosed basis, not measured optimal swing-up
frequencies. Full-amplitude sinusoids and bang-bang waves are a deliberate
proposal choice under the existing native[-1,1] action contract. No energy
controller, successful witness action, actor, critic or teacher parameter may
generate or select them. All32 native sequences must be simulated and scored
through the same paid model interface as other candidates.

Absolute episode time fixes each template's phase across replans. Store its
period, phase, waveform and family identity with the raw candidate. This gives
the *candidate representation* coherent continuation, while replanning can
still switch template or choose another channel. It does not impose a full
wave on the executed controller. The source must not silently introduce
state-dependent feedback or simulator calls while constructing a sequence.

Keep native nominal retention, Gaussian row0, scoring, paid mean verification
and the best-row-versus-mean champion rule unchanged. Keep constant nominal
tail filling unchanged in this first comparison, so it tests the32-slot
representation alone. If the existing mean mixes different phases, it remains
a separately simulated alternative; no unverified mean can displace a better
queried bank row. Do not simultaneously add a phase-conditioned mean, a new
terminal value, a commitment rule or a different objective. Those would be
separate mechanisms. Maintaining template phase does not guarantee selected
action phase consistency; inspect the actual executed switches.

## Matched comparison and freeze requirements

Compare against the strongest frozen independent proposal version available
before this assay: the original short knots or the block version, selected by
the closed preceding evidence and explicitly named in the new protocol. Do
not describe a comparison to an obsolete weaker version as a coverage victory.
Freeze that choice, complete source hashes, seeds/panel, criteria and all cost
caps before any new rollout. Their values are intentionally not chosen from
unseen active results in this draft.

Both arms use H144, T800 and the unchanged task/physics/cost/capture definitions,
matched reset and planner seeds, empty initial memory, K512+32+16=560 bank rows,
one separately queried mean per decision, and the same actual execution rule.
Consume the baseline broad-slot RNG draw matrix even if the deterministic
templates do not use it, preserving private Gaussian/fallback streams at the
same planner state. Subsequent states may naturally diverge. Source tests must
check array bounds, template phase continuity, exact32 slots, preserved other
channels/RNG endpoint and failed-query accounting without treating those tests
as scientific success.

For one complete episode with one planning iteration per decision, the physical
control upper bound is

```
561 * sum(r=1,...,800, min(144,r)) + 800 = 58,851,944.
```

Here561 includes560 candidates plus the paid mean, and800 counts executed
controls. A two-arm/two-root assay would therefore have the same235,407,776
control cap as the preceding temporal assay, **not a new available allowance**.
The final protocol must reserve and charge every qualification, failed call,
verification and evaluation, and record actual wall time/storage. No padding,
fall budget reuse, free rescue, additional rollout-based tuning or repeated
selection after negative results is allowed. There is no actor inference or
fit work in this discovery-only proposal; any added work needs explicit scope
and accounting before launch.

## Measurements, failure categories and limits

Retain every raw bank, verified mean, decision, model restore, full executed
trajectory and terminal event. Report all roots and all costs. The primary
discovery outcomes remain actual local approach by its deadline and complete
slow capture, alongside complete cost; lower truncated heuristic alone does
not count as useful successful experience. Save candidate-source and waveform
switch histories, model/live first-step agreement, nearest tip distance,
shoulder/elbow configuration, speed, energy and selected-score components.
Energy is a descriptive diagnostic, not a replacement objective or success gate.

Classify separately: no useful sampled prefixes (coverage); useful prefixes
misranked by the finite score (optimization/objective truncation); model/live
disagreement (prediction/restore); and correctly predicted plans repeatedly
lost under replanning (executed control). Complete outcomes decide whether
the acquired experience is actually useful. Passing this small assay would
not establish broad reliability, optimality, sample efficiency, a curriculum
effect, a global actor, or a successful bidirectional GPS loop.

The weakest needed assumptions are that this native system admits capture
within the unchanged8 s episode on relevant starts; at least some useful
control sequences are represented or can be refined by the finite channels;
the qualified model/restore is accurate over evaluated continuations; and any
generalization beyond the fixed starts is separately supported. Small-angle
periods and a capacity witness do not prove the waveform-coverage assumption.
An actor cannot learn an ending from task dynamics or a failed prefix alone;
actor-owned complete return learning and earned/revocable guidance remain
later gates. Stop at the first unsupported link and retain a negative result.
