"""Training-seed blocks must not reuse reset states across independent runs."""
import pytest

from scripts.training import train_walker_mdgps as training


def test_reject_overlapping_training_seed_blocks_before_start(tmp_path, monkeypatch):
    def must_not_build_actor(seed):
        pytest.fail("overlapping seed blocks reached actor initialization")

    monkeypatch.setattr(training, "_new_actor", must_not_build_actor)
    cfg = training.TrainConfig(iterations=101, episodes_per_iter=100)
    with pytest.raises(ValueError, match="10,000"):
        training.main(tmp_path / "invalid", cfg)
    assert not (tmp_path / "invalid").exists()
