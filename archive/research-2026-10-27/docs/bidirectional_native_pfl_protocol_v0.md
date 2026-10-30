# Native collocated-PFL feasibility reference

This finite development diagnostic tests a classical analytic controller on the
unchanged Acrobot task. It is separate from independent MPPI acquisition and the
research algorithm. All traces are excluded from teacher data. A successful
reference would show a feasible controller under these conditions, not MPPI
coverage, policy learning, novelty or sample efficiency. A negative is not a task
infeasibility proof.

The previous energy reference used a direct elbow-speed/power heuristic. The
present law uses collocated partial feedback linearization and shoulder-speed
energy feedback, described in [MIT's Acrobot notes](https://underactuated.csail.mit.edu/acrobot.html)
and related to [Spong's swing-up work](https://doi.org/10.1109/37.341864).
Those references provide no guarantee for this plant, saturation and deadline.

## Fixed mechanics and control

Preserve the compiled XML capsule masses/inertias, gravity, damping.05 perjoint,
positive elbow gear2, .01-second RK4,800 controls, original task costs and1000
capture-failure penalty. Starts remain near hanging; local approach is distance
<=.5 by control600; capture requires distance<=.2 and joint-speed norm<=.5
throughout the final100 boundaries. True terminal cost occurs once.

Use the existing compiled-model parameter extractor, native mass matrix, energy
and upright linearization. Native q1=0 is upright; q2 is relative elbow angle.
With M*qdd+h=[0;gear*u], define c=m2*l1*c2*sin(q2),
Cw=[-c*(2*w1*w2+w2^2),c*w1^2],
G=-g*[(m1*c1+m2*l1)*sin(q1)+m2*c2*sin(q1+q2),m2*c2*sin(q1+q2)],
and h=Cw+G+D*w. Then

```text
v = -k1*q2_unwrapped - k2*w2 + k3*w1*(E-E_upright)
S = M22 - M21*M12/M11
tau_requested = S*v + h2 - (M21/M11)*h1
u = clip(tau_requested/gear, -1, 1)
```

The exact continuous cancellation holds only without saturation. After clipping,
q2dd-v=(gear*u-tau_requested)/S. Preserve unwrapped q2 in the PFL posture term.
There is no excitation, noise, online gain fitting or model search in this law.

The four fixed gain triples are (1,1,.1),(1,1,1),(4,2,.1),(4,2,1). In the small
upright region, switch to the existing native continuous LQR using wrapped angle
error only: strict angle norm<.3 AND velocity norm<2. Q=diag(10,10,1,1), R=.1.
Solve one CARE equation, check its residual and closed-loop eigenvalues, then
share that fixed gain across four controller identities. Continuous local
stability does not certify sampled, clipped task completion.

## Native qualification before episodes

Read compiled parameters once and preserve the raw model arrays. Before CARE or
any integration, inspect eight fixed (q1,q2,w1,w2) states:

```text
(pi,0,0,0); (pi+.05,.02,.04,-.03); (0,0,0,0);
(.1,-.15,.2,-.3); (.4,-.7,3,-2); (2.4,1.2,-4,1);
(-1.3,2*pi+.4,1.5,-3); (1.3,-2*pi-.4,-1.5,3).
```

At each, inspect actions -1,+1 and the clipped PFL action using gain index
state_index modulo4. Each of these24 cases performs one scratch resetData, one
zero-time mj_forward and one mj_fullM expansion. No integration or rollout is
permitted here. Log intent before every engine call and completion afterward.
Compare analytic/native mass, bias=Cw+G, passive=-D*w, actuator torque, qacc,
Schur elimination, clipping residual and energy. Require absolute1e-9/relative
1e-10 for mass/bias/passive/torque and absolute1e-8/relative1e-10 for acceleration,
elimination, clipping and energy. Any failure stops before CARE/live episodes;
do not adjust tolerances after observing the result.

The installed3.6.0 static header specifies mj_fullM(model,destination,packedM).
This inspection qualification does not establish full-trajectory numerical
parity, controller success or a formal safety property.

## Actual outcome gates and finite work

Run all four gains on the same two fresh ordinary roots210000/210001,800 controls
each. Only a gain passing BOTH original local and capture criteria on BOTH roots
gets the same eight fresh roots211000--211007; all eight must pass. Select the
eligible gain with lowest complete mean cost, stable index ties. Then evaluate
that fixed controller on32 new roots212000--212031; require at least29 to pass
both criteria. Do not change gains, switching region, reward, deadline, roots,
selection metric or budget. Failure closes this version; later hypotheses get
new versioned protocols.

The cap is57,600 integrated native controls:6400 initial +25600 selection
+25600 final. All actual controls use one counted native query executor with a
shared before-call cap. Failures and rejected queries retain their charges and
raw evidence. Record requested/completed controls, physics frames/substeps,
state/warm-start restoration, component costs, controller action arithmetic,
branch and saturation, wall time and hardware. There are zero actor forwards,
Adam steps or training updates. Zero-time work is counted separately: at most24
qualification forwards,24 mass expansions, plus reset/restore forwards for at
most72 actual episodes (144 more forwards, total168). Parameter read, model/data
construction, one CARE solve and four controller constructions are recorded.

Freeze source/config/protocol and run accountable pure checks before launch.
An independent raw arithmetic/ownership audit is required for the final verdict.
Do not use a low analytic cost, an energy crossing, local approach alone or a
solver's status in place of the executed complete task outcome.
