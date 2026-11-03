from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.gps.mppi_gps import similarity_atlas as atlas


def fixture_trace(count=7):
    states = np.zeros((count + 1, 19))
    states[:, 0] = 0.008 * np.arange(count + 1)
    states[:, 2] = 1.2
    states[:, 3:10] = np.sin(np.arange(count + 1)[:, None] * np.arange(1, 8) * 0.1)
    contacts = np.zeros((count, 2), dtype=bool)
    contacts[[1, 3, 6], 0] = True
    return dict(
        tag="synthetic_contact_flicker_and_fall",
        states=states,
        actions=np.arange(count * 6).reshape(count, 6) / max(1, count * 6),
        contacts=contacts,
        forces=contacts.astype(float) * 10,
        integrations=np.zeros((count + 1, 91)),
        warmstarts=np.zeros((count + 1, 9)),
        done=np.r_[np.zeros(count - 1, dtype=bool), True],
    )


def test_raw_spikes_dropouts_and_partial_intervals_are_not_debounced():
    values = fixture_trace()
    rows = atlas.stride_segments(values["contacts"])
    assert [(r["start_state_boundary"], r["stop_state_boundary"]) for r in rows] == [
        (0, 2),
        (2, 4),
        (4, 7),
        (7, 7),
    ]
    assert [r["kind"] for r in rows] == [
        "initial_partial",
        "complete_onset_interval",
        "complete_onset_interval",
        "terminal_partial",
    ]
    assert sum(row["controls"] for row in rows) == 7
    assert rows[0]["right_onset_contact_row"] is None
    assert [r["right_onset_contact_row"] for r in rows[1:]] == [1, 3, 6]


@pytest.mark.parametrize("right", [[False] * 9, [True] * 9, []])
def test_nonperiodic_constant_contact_has_no_invented_stride(right):
    contacts = np.zeros((len(right), 2), dtype=bool)
    contacts[:, 0] = right
    rows = atlas.stride_segments(contacts)
    assert len(rows) == 1
    assert rows[0]["kind"] == "unsegmented_partial"
    assert rows[0]["controls"] == len(right)


def test_initial_contact_is_not_an_observed_false_to_true_transition():
    contacts = np.array([[True, False], [True, False], [False, True], [True, True]])
    rows = atlas.stride_segments(contacts)
    assert rows[0]["stop_state_boundary"] == 4
    assert sum(r["right_onset_contact_row"] is not None for r in rows) == 1


def test_partial_terminal_frame_has_exact_pts_and_only_explicit_display_holds():
    frames = atlas.frame_schedule(7, terminal_hold_frames=2)
    assert [r["state_index"] for r in frames] == [0, 4, 7, 7, 7]
    assert [r["presentation_tick_125hz"] for r in frames] == [0, 4, 7, 11, 15]
    assert [r["terminal_display_hold"] for r in frames] == [False, False, False, True, True]
    assert frames[2]["physical_time_seconds"] == pytest.approx(0.056)
    assert all(r["physical_time_seconds"] == pytest.approx(0.056) for r in frames[2:])
    assert atlas.frame_schedule(8)[-1]["state_index"] == 8
    assert len(atlas.frame_schedule(8)) == 3


@pytest.mark.parametrize("count", [-1, 1001, True, 1.5])
def test_frame_roster_and_timing_bounds(count):
    with pytest.raises(ValueError):
        atlas.frame_schedule(count)


def test_trace_rejects_post_fall_rows_and_missing_contact_measurements():
    value = fixture_trace()
    value["done"][2] = True
    with pytest.raises(ValueError, match="after an actual fall"):
        atlas._trace(value)
    value = fixture_trace()
    value["contacts"] = np.zeros((7, 0), dtype=bool)
    with pytest.raises(ValueError, match="contacts"):
        atlas._trace(value)
    value = fixture_trace()
    value["forces"] = np.zeros((7, 2, 0))
    with pytest.raises(ValueError, match="missing evidence"):
        atlas._trace(value)
    value = fixture_trace()
    value["states"][3, 0] += 0.002
    with pytest.raises(ValueError, match="timestamps"):
        atlas._trace(value)


def test_module_import_is_lazy_for_native_simulator_and_policy():
    source = (
        "import sys; from src.gps.mppi_gps import similarity_atlas; "
        "assert 'mujoco' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", source], check=True, timeout=10)


def test_budget_stop_keeps_an_emergency_receipt(tmp_path):
    budget = atlas._Budget(tmp_path, 10, 5 * 1024**2, 2 * 1024**3)
    budget.deadline = 0
    with pytest.raises(RuntimeError, match="wall/RSS"):
        budget.write(tmp_path / "not_written", b"x")
    budget.closing_json(tmp_path / "failure.json", dict(passed=False, preserved=True))
    assert json.loads((tmp_path / "failure.json").read_text())["preserved"] is True
    assert not (tmp_path / "not_written").exists()


def test_encoder_hash_gate_precedes_any_execution(tmp_path):
    path = tmp_path / "encoder"
    path.write_bytes(b"This is deliberately not an executable")
    with pytest.raises(ValueError, match="ffmpeg bytes"):
        atlas.encoder_binding(path, "0" * 64)


def test_real_encoder_functional_preflight_is_tiny_and_preserves_pts():
    executable = shutil.which("ffmpeg")
    if executable is None:
        pytest.skip("Scientific launch requires a pinned encoder")
    binding = atlas.encoder_binding(executable)
    if not (Path(binding["path"]).parent / "ffprobe").is_file():
        pytest.skip("Scientific launch requires the pinned encoder's ffprobe family")
    result = atlas.encoder_functional_probe(binding)
    assert result["passed"] is True and result["calibration_frames"] == 3
    assert result["observed_presentation_times_seconds"] == [0, 0.032, 0.056]
    assert result["temporary_output_bytes"] < 2 * 1024**2
    assert result["temporary_output_removed"] is True
    assert result["native_controls"] == result["neural_rows"] == result["fitting_updates"] == 0


def test_offline_schematic_uses_pinned_geometry_without_advancing_physics():
    import mujoco

    path = Path(__file__).resolve().parents[1] / "assets/walker2d.xml"
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    size = mujoco.mj_stateSize(model, spec)
    snapshots, states = [], []
    for index in range(2):
        data.time = index * atlas.DT
        data.qpos[:] = model.qpos0
        data.qpos[3] += 0.6 * index
        integration = np.empty(size)
        mujoco.mj_getState(model, data, integration, spec)
        snapshots.append(integration)
        states.append(np.r_[data.time, data.qpos, data.qvel])
    arrays = dict(states=np.array(states), integrations=np.array(snapshots))
    original = {key: value.copy() for key, value in arrays.items()}
    backend = atlas._OfflineFrames(path, "kinematic_schematic")
    try:
        first = backend.frame(arrays, 0, "synthetic_geometry")
        changed = backend.frame(arrays, 1, "synthetic_geometry")
        assert first.shape == (240, 320, 3) and first.dtype == np.uint8
        assert not np.array_equal(first[40:], changed[40:])
        assert backend.work == dict(
            offline_restores=2,
            offline_mj_forward_calls=0,
            offline_position_forward_calls=2,
        )
        for key in original:
            np.testing.assert_array_equal(arrays[key], original[key])
        contradictory = dict(arrays, states=arrays["states"].copy())
        contradictory["states"][0, 1] += 0.1
        with pytest.raises(ValueError, match="physical state disagree"):
            backend.frame(contradictory, 0, "contradictory")
    finally:
        backend.close()


def test_small_real_encoder_atlas_preserves_partial_pts_and_actual_arrays(tmp_path, monkeypatch):
    executable = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    if executable is None or ffprobe is None:
        pytest.skip("Local encoder tools unavailable; scientific launch requires pinned preflight")

    class KnownFrames:
        def __init__(self, xml, mode):
            self.work = dict(
                offline_restores=0, offline_mj_forward_calls=0, offline_position_forward_calls=0
            )

        def frame(self, arrays, index, tag):
            self.work["offline_restores"] += 1
            return np.full((240, 320, 3), 30 * index, dtype=np.uint8)

        def close(self):
            pass

    monkeypatch.setattr(atlas, "_OfflineFrames", KnownFrames)
    monkeypatch.setenv("MPLCONFIGDIR", str(tmp_path / "mpl"))
    value = fixture_trace()
    encoder = atlas.encoder_binding(executable)
    model = Path(__file__).resolve().parents[1] / "assets/walker2d.xml"
    out = tmp_path / "atlas"
    result = atlas.render_atlas(
        [value],
        out,
        xml_path=model,
        mode="kinematic_schematic",
        ffmpeg_path=encoder["path"],
        ffmpeg_sha256=encoder["sha256"],
        wall_seconds=90,
    )
    assert result["passed"] is True and result["frames"] == 3
    assert result["work"]["native_controls"] == 0
    assert result["traces"][0]["actual_states"] == 8
    assert (
        "KINEMATIC" not in (out / "index.html").read_text()
        or "schematic" in (out / "index.html").read_text()
    )
    with np.load(out / "trace_000/actual_scalars.npz") as saved:
        np.testing.assert_array_equal(saved["contacts"], value["contacts"])
        np.testing.assert_array_equal(saved["states"], value["states"])
        assert len(saved["actions"]) == 7
    probe = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "frame=pts_time",
            "-of",
            "json",
            str(out / "trace_000/video.mp4"),
        ],
        capture_output=True,
        check=True,
        timeout=10,
    )
    pts = [float(row["pts_time"]) for row in json.loads(probe.stdout)["frames"]]
    np.testing.assert_allclose(pts, [0, 0.032, 0.056], rtol=0, atol=1e-9)
    assert (out / "trace_000/motion_contacts_controls.png").stat().st_size > 0
    with pytest.raises(FileExistsError):
        atlas.render_atlas(
            [value],
            out,
            xml_path=model,
            mode="kinematic_schematic",
            ffmpeg_path=encoder["path"],
            ffmpeg_sha256=encoder["sha256"],
        )
