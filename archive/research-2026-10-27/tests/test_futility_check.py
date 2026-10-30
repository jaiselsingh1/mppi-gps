"""Protocol rule 8: flat arms are futile, rising or completing arms are not."""
from scripts.diagnostics.futility_check import verdict


def _run(alive_seconds, completions=0):
    return [{"iteration": 4 + 5 * k, "eval_dev": {"mean_steps": a / 0.008, "survival_rate": completions / 10,
                                                   "episodes": 10}} for k, a in enumerate(alive_seconds)]


def test_rule_separates_flat_from_rising_arms():
    flat = [_run([1.6, 1.3, 1.2, 1.1, 1.7, 1.4])] * 3  # Step 15's student
    rising = [_run([1.2, 1.3, 1.6, 1.4, 1.6, 2.1])] * 3  # MPPI-GPS, Steps 10/12
    assert verdict(flat)["futile"]
    assert not verdict(rising)["futile"]


def test_completions_or_too_few_evaluations_keep_an_arm_alive():
    assert not verdict([_run([1.2, 1.2, 1.2, 1.2, 1.2, 1.2], completions=3)])["futile"]
    assert not verdict([_run([1.2, 1.2, 1.2, 1.2, 1.2])])["decided"]
