# Independent MPPI and a coherent global policy

This implementation tests independent control-space discovery, global policy
learning, and outcome-gated policy agreement in local search. It does not yet
establish successful Acrobot capture, standalone Walker walking, improved
exploration, or sample efficiency. The first real horizon/task screen failed:
all four horizons had zero local approaches and zero captures on two resets
each. The learned scalar Ending pilot established a narrower partial mechanism;
the physical task remains unqualified. See [recorded findings](bidirectional_findings_2026-10-27.md).
Numerical backend qualification is a separate implementation check.

## Task and complete credit

The existing Acrobot XML is retained: 0.01-second RK4 controls, one motor command
in `[-1,1]`, and gear 2. A private reset stream places the shoulder near hanging
down and the elbow near straight, with bounded position and velocity variation.
The default episode contains 800 controls. Useful swing-up is entry within
0.5 m of the target by control 600. Capture requires target distance at most
0.2 m and joint-speed norm at most 0.5 rad/s throughout the final 100 boundaries.

The explicit `acrobot-capture-v2-post-action-fk` task computes geometry from
post-action full states. This repairs the diagnosed CPU/Warp RK4 sensor-timing
difference without changing the original cost algebra or weights. Stage cost
retains height/centering, gated velocity and excessive-speed terms. Complete
credit adds the original terminal cost once at the true episode end and a
1000-point charge if capture fails. An endpoint cost inside a shorter model
rollout is labelled a planner heuristic. It is not a measured ending outcome.
Legacy sensor and cost outputs remain in raw query records as diagnostics.

## The loop

1. Qualify the runtime and task before learning. The declared horizon roster
   compares independent controllers using the same reset streams. A candidate
   pair needs competent short-horizon swing-up, poor short-horizon capture, and
   successful longer-horizon capture. Fresh confirmation also gives the short
   horizon matched planning work. This can reject a horizon explanation when
   more short-horizon search solves the problem. A failed pair is retained;
   the objective is not changed to manufacture a gap.
2. Acquire complete, executed independent MPPI trajectories and retain failures.
   The planner keeps a private nominal and RNG. Its bank combines its own
   bounded Gaussian proposals, independently sampled smooth controls, and
   nearest-neighbor executed trajectory memory. The weighted mean is queried
   and compared with the best bank member before execution. Actor outputs never
   seed this proposal bank or overwrite its nominal.
3. Partition successful discovery trajectories by the direction of the
   unwrapped shoulder at first target approach. A family-conditioned bounded
   Gaussian actor compresses executed teacher actions as explicitly supervised
   auxiliary labels. These labels evaluate the teacher's continuation. They
   are never relabelled as actor-owned returns.
4. Improve the actor using its own complete stochastic continuations from both
   ordinary resets and acquired handovers. Repeated trials supply Monte Carlo
   cost-to-go with a leave-one-out context/family/horizon baseline. A clipped
   policy-gradient update proposes a change; fresh deterministic and stochastic
   ordinary-root executions decide promotion. The persistent family is part of the deployed
   controller identity. Neither actor fitting loss nor stochastic return alone
   certifies the deterministic controller.
5. Start agreement influence at zero. Successful complete actor-only handovers
   qualify bounded state/time support. A provisional weight from the fixed
   0.1, 1, 10 ladder then faces paired *executed replanning* trials. Promotion
   requires lower complete cost without losing captures. Actor/task changes,
   unsupported states, numerical errors and failed qualification remove
   influence. Qualified agreement reranks immutable independent proposals and
   queries its own mean; it does not modify the independent planner update.
6. Compare learning from independent teacher data against learning with the
   qualified bidirectional teacher. Both branches receive the same initial
   corpus and memory. New teacher trajectories describe the controller actually
   executed. Actor-only and executed-replanning panels are distinct outcomes.

The discovery mechanism is independent model search plus executed trajectory
memory. The learner generalizes and compresses discovered feedback experience;
it cannot invent evidence for an unseen successful ending. Coverage of useful
trajectories, interpolation to actor-visited states, and the bounded optimizer's
ability to find useful updates remain empirical assumptions. Agreement alone
does not establish that better hypothetical plans improve executed control.

The actor objective is explicit. For a complete stochastic behavior episode
`i` owned by frozen actor `theta_b`, let `G_it = sum_{k=t}^{T-1} c_ik`, with
the true terminal and capture-failure costs included in the final `c` exactly
once. A context consists of initial snapshot identity, family and remaining
horizon. Repeated independent executions of that same context supply the
leave-one-trajectory-out baseline `b_it`; the cost advantage is
`A_it = (G_it - b_it) / 800` for the full Acrobot task. The actor minimizes

```text
mean(max(r_it * A_it, clip(r_it, 0.8, 1.2) * A_it))
  + 0.1 * mean(||tanh(mu_theta(o,z)) - u_teacher||^2)
r_it = pi_theta(u_it | o_it,z) / pi_theta_b(u_it | o_it,z)
```

The maximum, rather than the reward-maximizing minimum, implements a clipped
cost objective. Squashed-Gaussian likelihoods use the saved pre-tanh latent
and Jacobian correction. Behavior hashes, complete query references and
original likelihoods are checked before fitting. Four epochs propose an
update; clipping does not turn old measurements into returns of the changed
actor. The auxiliary term uses only successful executed teacher trajectories
in the chosen family. Its targets describe the teacher's actions; its measured
benefit must be tested on the resulting student. There is no terminal critic,
Bellman target, or actor-derived proposal channel in this version.

## Explicit development revision

`configs/bidirectional_acrobot_v0.json` is an initial development candidate,
not a completed frozen scientific comparison. Its original rollback and
sequential-weight behavior remain available. The new opt-in
`configs/bidirectional_acrobot_v1.json` makes two changes motivated by the
retained scalar pilot failures; it changes no Acrobot physics, reward, reset,
capture criterion, or existing evidence.

The training learner and deployed learner retain separate complete parameters,
Adam state, private RNG streams and family identities. An unsuccessful
deployment proposal can still become the next training learner. Its next
Monte Carlo targets must come from fresh executions of that exact training
actor. Temporary teaching-family updates share the same parent, behavior data
and optimizer work. Complete development outcomes select the next training
family; the separately deployed actor changes only when fresh deterministic
and stochastic complete-root panels pass. Families remain fixed within an
episode. Guidance and final standalone evaluation use the deployed actor.

Agreement settings 0.1, 1 and 10 receive isolated provisional tests from the
same actor-suffix eligibility packet. A failed candidate does not veto the
other settings. Successful candidates are ranked by capture, then complete
cost, then smaller weight. The selected setting faces an additional fresh
paired executed-controller confirmation before deployment. Candidate selection,
confirmation, handoffs, fitting and evaluation are all charged. If no setting
passes confirmation, influence stays zero. The original sequential protocol is unchanged.

The matched child assay separately binds the guiding actor and the parent
that owns the Monte Carlo data. Both children retain the identical initial
independent discovery corpus, including when the raw teacher obtains no new
success. Fixed evenly spaced row selection gives equal auxiliary batch sizes
over each common-plus-new corpus. Fresh complete actor-only outcomes compare
the resulting children; a low fit loss is not its selection criterion.
This revision is implemented as software. It has not produced an ordinary-start
Acrobot learning result.

## Reference, evidence and costs

Acquisition also runs a separately counted analytic reference on the same reset
seeds. It uses exact two-link energy shaping with declared deterministic
excitation, switching near upright to saturated continuous-time LQR. Masses,
COMs, inertias, gravity, damping and actuator gear come from the loaded model;
action arithmetic performs no simulator search. This is a fixed reference,
not a claim of the strongest possible analytic controller. Its trajectories
and actions are explicitly excluded from discovery qualification and all
teacher auxiliary data, even if the reference succeeds.

Each model/live call passes through `QueryExecutor`. A fsynced request precedes
the call; compressed arrays retain pre/post states, actions, observations,
components, ownership and context. Live snapshots retain exact warm starts,
elapsed time and capture history. Model endpoint zero warm starts are labelled
interventions rather than exact recovered solver states. Requested horizon
reservations, actual returned controls, skipped cutoff slots, physics frames,
failed calls and unknown interrupted work are separate quantities. Neural work,
fitting, verification, qualification, evaluation, and wall time also belong in
the cost report. Reused evidence is not new discovery.

Work is staged with frozen source/configuration and explicit per-stage rosters;
there is no imposed overall four-hour cap. The initial development horizon
roster is versioned to two paired roots per horizon after paid production
measurement of full raw storage and throughput. The earlier eight-root value
was an implementation default. Two roots only screen candidate conditions;
the separate 32-root confirmation is required before declaring the task gap
qualified. Storage/time forecasts are estimates, not measured controller
performance or a reason to discard failed attempts.

## CLI stages

Run from the repository environment, replacing paths with new stage directories:

```sh
.venv/bin/python -m scripts.diagnostics.run_bidirectional_tests --out NEW_TEST_RUN
.venv/bin/python -m scripts.diagnostics.run_learned_ending_pilot --config configs/bidirectional_ending_pilot_v2.json --out NEW_ENDING_RUN
python -m scripts.training.train_bidirectional_gps --stage validate --config CONFIG --out NEW_VALIDATION
python -m scripts.training.train_bidirectional_gps --stage qualify --config CONFIG --out NEW_QUALIFICATION --backend-qualification BACKEND_REPORT
python -m scripts.training.train_bidirectional_gps --stage acquire --config CONFIG --out NEW_ACQUISITION --qualification QUALIFICATION_JSON --backend-qualification BACKEND_REPORT
python -m scripts.training.train_bidirectional_gps --stage develop --config CONFIG --out NEW_DEVELOPMENT --corpus ACQUISITION_DIR --qualification QUALIFICATION_JSON --backend-qualification BACKEND_REPORT
python -m scripts.training.train_bidirectional_gps --stage compare --config FROZEN_COMPARISON_CONFIG --out NEW_COMPARISON --corpus ACQUISITION_DIR --qualification QUALIFICATION_JSON --backend-qualification BACKEND_REPORT
python -m scripts.training.train_bidirectional_gps --stage report --out CLOSED_STAGE
```

The Ending pilot exits with code 2 when its declared scientific gate fails;
that is a retained negative result. It uses the production actor, own-return
learner, protected search and trust interfaces. It does not substitute for the
physical task, a three-seed comparison, or next-generation learning. Independent
offline audits add no physics or actor inference:

```sh
.venv/bin/python -m src.gps.bidirectional.ending_audit CLOSED_ENDING_RUN --out NEW_AUDIT_JSON
.venv/bin/python -m src.gps.bidirectional.audit CLOSED_ACROBOT_RUN --out NEW_AUDIT_JSON
```

Warp stages require passed evidence binding the current task and backend source.
Outputs are create-only; interruption evidence remains charged and a resumed
stage uses a new directory with parent provenance. Stage completion and passing
software tests are not scientific success. Acrobot must establish the relevant
links before a separately qualified Walker investigation. A standalone policy
result requires actor-only execution; an actor-free value-guided MPPI pilot
would establish only a mechanism test. An efficiency claim additionally needs
strong paid-cost comparisons, including trajectory memory and any relevant
learned-value MPC control; a same-K performance gain alone is insufficient.
