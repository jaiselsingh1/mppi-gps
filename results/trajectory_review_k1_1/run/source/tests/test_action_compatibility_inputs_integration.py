"""Synthetic extractor-to-orchestrator contracts, without actor inference."""

from src.gps.mppi_gps.experiments.action_compatibility import ancestry_id, group_scores


def test_extractor_ancestry_dict_and_execution_string_share_the_same_cluster():
    saved = {"ancestry": {"reset_seed": 17, "reset_cluster": "native-reset-17", "scope": "saved"}}
    executed = {"ancestry": "native-reset-17"}
    assert ancestry_id(saved) == ancestry_id(executed) == "native-reset-17"


def test_group_uncertainty_clusters_paired_trajectories_instead_of_frames():
    names = ["untrained_0", "untrained_1", "untrained_2", "h4_parent"]
    records = []
    for seed in (17, 18):
        for arm in ("independent", "fixed"):
            records.append(
                {
                    "trajectory": {
                        "id": f"{arm}_{seed}",
                        "group": "synthetic_paired_group",
                        "steps": 1000,
                        "ancestry": {"reset_seed": seed, "reset_cluster": f"native-reset-{seed}"},
                    },
                    "policies": {
                        name: {"mse": value}
                        for name, value in zip(names, (0.5, 0.6, 0.7, 0.4), strict=True)
                    },
                }
            )
    result = group_scores(records, names, {"bootstrap_seed": 3, "bootstrap_draws": 20})
    group = result["synthetic_paired_group"]
    assert group["trajectories"] == 4
    assert group["ancestries"] == 2
    assert group["executed_rows"] == 4000
    assert group["equal_trajectory_mean"] == [0.5, 0.6, 0.7, 0.4]
    assert group["contrasts"]["h4_parent_minus_mean_untrained"]["delta"] < 0
