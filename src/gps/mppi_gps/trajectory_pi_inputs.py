"""Create-only frozen inputs for the bounded full-trajectory PI comparison.

No teacher acquisition, policy forward or fitting occurs here. Recorded parent
packages are copied exactly; the historical Adam/private RNG states are retained.
New-study global RNG envelopes are declared separately from unavailable history.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from .provenance import sha, write
from .trajectory_pi_student import freeze_legacy_parent, observations


def binding(path: Path) -> dict:
    return dict(path=str(path.resolve()), bytes=path.stat().st_size, sha256=sha(path))


def prepare(config_path: Path, directory: Path) -> dict:
    config = json.loads(config_path.read_text())
    if config["schema"] != "trajectory-pi-p1-1" or not config["launch_ready"]:
        raise ValueError("A committed frozen P1 configuration is required")
    directory.mkdir(parents=True, exist_ok=False)
    originals = directory / "originals"
    originals.mkdir()
    teachers, parents = [], []
    for spec in config["teachers"]:
        source_spec = spec["episode"]
        source = Path(source_spec["path"])
        if source.stat().st_size != source_spec["bytes"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Recorded teacher source binding differs")
        shutil.copyfile(source, originals / f"teacher_{spec['slot']}.npz")
        copied = originals / f"teacher_{spec['slot']}.npz"
        if sha(copied) != source_spec["sha256"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Teacher source changed during preparation")
        with np.load(source, allow_pickle=False) as raw:
            values = dict(
                states=np.concatenate((raw["states"], raw["terminal_state"][None])),
                warmstarts=np.concatenate((raw["warmstarts"], raw["terminal_warmstarts"][None])),
                integrations=np.concatenate(
                    (raw["integrations"], raw["terminal_integrations"][None])
                ),
                observations=raw["observations"].copy(),
                actions=raw["actions"].copy(),
                costs=raw["costs"].copy(),
            )
        if (
            values["states"].shape != (1001, 19)
            or values["actions"].shape != (1000, 6)
            or values["warmstarts"].shape != (1001, 9)
            or values["integrations"].shape != (1001, 91)
            or any(
                value.dtype != np.float64 or not np.isfinite(value).all()
                for value in values.values()
            )
        ):
            raise ValueError("Initial teacher must retain complete finite float64 contexts")
        if not np.array_equal(observations(values["states"][:-1]), values["observations"]):
            raise ValueError("Initial teacher pre-action observations differ")
        healthy = (
            (values["states"][1:, 2] > 0.8)
            & (values["states"][1:, 2] < 2)
            & (np.abs(values["states"][1:, 3]) < 1)
        )
        mae = float(np.mean(np.abs(values["states"][1:, 10] - 1.5)))
        if not healthy.all() or mae > 0.3:
            raise ValueError("Initial teaching sequence is not complete competent walking")
        target = directory / spec["prepared_path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            np.savez_compressed(stream, **values)
        teachers.append(
            dict(
                slot=spec["slot"],
                file_relative=spec["prepared_path"],
                source=binding(source),
                complete_cost=float(values["costs"].sum()),
                walking=True,
                velocity_mae=mae,
            )
        )
    normalizations = []
    for spec in config["parents"]:
        source_spec = spec["checkpoint"]
        source = Path(source_spec["path"])
        if source.stat().st_size != source_spec["bytes"] or sha(source) != source_spec["sha256"]:
            raise ValueError("Recorded incoming policy/Adam package differs")
        target = directory / spec["prepared_directory"]
        policy, state, envelope = freeze_legacy_parent(
            source,
            target,
            global_seed=spec["new_global_envelope_seed"],
            metadata=dict(study="trajectory-pi-p1-1", stream=spec["stream"], parent_refitted=False),
        )
        if state["updates"] != 2000 or state["fit_updates"] != 2000:
            raise ValueError("Expected the recorded W2 2,000-update endpoint")
        normalizations.append(policy.normalization)
        parents.append(
            dict(
                stream=spec["stream"],
                directory=spec["prepared_directory"],
                source=binding(source),
                policy_hash=policy.hash(),
                optimizer_fingerprint=envelope["optimizer_fingerprint"],
                global_rng_fingerprint=envelope["global_rng_fingerprint"],
            )
        )
    if (
        len(parents) != 3
        or len(teachers) != 2
        or any(
            not np.array_equal(item[key], normalizations[0][key])
            for item in normalizations
            for key in ("mean", "std")
        )
    ):
        raise ValueError("Three matched W2 normalizations and two teachers required")
    roster = [
        dict(path=str(path.relative_to(directory)), bytes=path.stat().st_size, sha256=sha(path))
        for path in sorted(directory.rglob("*"))
        if path.is_file()
    ]
    result = dict(
        schema="trajectory-pi-prepared-1",
        config_sha256=sha(config_path),
        parents=parents,
        teachers=teachers,
        files=roster,
        normalization_inherited=True,
        parent_refits=0,
        historical_global_rng_available=False,
        new_study_globals_frozen=True,
        preparation_work=dict(
            checkpoint_loads=3,
            policy_constructions=3,
            neural_rows=0,
            native_controls=0,
            optimizer_steps=0,
        ),
    )
    write(directory / "manifest.json", result)
    return result
