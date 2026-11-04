The saved controller has two distinct kinds of continuation. Within one
acquisition episode, MPPI shifts and carries the updated H48 plan after each
executed action. At the start of every acquisition episode, a newly constructed
planner resets that plan to exactly zero. It performs two initial refinements
at the same complete simulator context before executing an action. The second
refinement starts from the first refinement's unshifted updated plan, with the
next random draw. Comparing those two banks therefore combines changed plan
initialization with changed samples. It does not isolate the benefit of a warm
nominal under common random numbers.

Acquisition uses eight fixed complete restart contexts from the prepared
package throughout the loop. The updated actor changes the positive-alpha
candidate weights, while the independent arm retains alpha zero. The
standalone evaluation states at controls0,50,100 are used only for independent
scratch-MPPI disagreement probes. The probes' states and queried actions do not
enter acquisition or fitting. The retained fitting corpus contains the latest
competent300-control acquisition episode for each fixed slot, with fallback
to prior competent experience. This is behavioral coupling on fixed starts;
it is not corrective acquisition at the student's current visited states.

One minimal future causal test would bind one complete restart context, fixed
actor/alpha and private RNG state, then compare one H48/K256 refinement from
zero versus a predefined valid nonzero nominal, sharing innovations and native
runtime. It would preserve full banks and executed continuation for both.
Nothing of that kind is launched by this review.

These are source-level findings. The separate saved-bank table reports
descriptive cost-minimum, candidate-survival, ESS and action changes. Candidate
statistics are not executed-policy outcomes; repeated independent solves on a
fixed stream-slot seed are not independent task replications.
