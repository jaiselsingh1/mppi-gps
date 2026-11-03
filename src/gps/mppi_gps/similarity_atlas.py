"""J1 saved-trace stride atlas and explicitly timed offline video.

Nothing here advances physics or constructs a policy. Contacts are the supplied
post-control measurements, never inferred from zero-width model sensors.
"""

from __future__ import annotations

import csv
import hashlib
import html
import io
import json
import os
import resource
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

MAX_WALL = 7200
MAX_BYTES = 2 * 1024**3
MAX_RSS = 2 * 1024**3
MAX_TRACES = 224
MAX_FRAMES = 64000
DT = 0.008


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(value: Any) -> dict:
    array = np.ascontiguousarray(value)
    return dict(
        shape=list(array.shape),
        dtype=array.dtype.str,
        bytes=array.nbytes,
        sha256=hashlib.sha256(array.tobytes()).hexdigest(),
    )


def stride_segments(contacts: Any, *, dt: float = DT) -> list[dict]:
    """Partition actual controls at observed right-foot false→true boundaries.

    Row j is contact measured *after* control j, at state boundary j+1.
    The unknown contact before the first row is never assumed false. Empty
    terminal intervals preserve an onset on the final observed boundary.
    """
    array = np.asarray(contacts)
    if array.ndim != 2 or array.shape[1] != 2 or array.dtype.kind != "b":
        raise ValueError("contacts must be an actual (N,2) boolean array")
    if not np.isfinite(dt) or dt <= 0:
        raise ValueError("Positive finite control duration required")
    count = len(array)
    onsets = (np.flatnonzero(~array[:-1, 0] & array[1:, 0]) + 2).tolist()
    boundaries = [0, *onsets, count]
    rows = []
    for index, (start, stop) in enumerate(zip(boundaries[:-1], boundaries[1:], strict=True)):
        kind = "unsegmented_partial" if not onsets else "complete_onset_interval"
        if onsets and index == 0:
            kind = "initial_partial"
        elif onsets and index == len(boundaries) - 2:
            kind = "terminal_partial"
        rows.append(
            dict(
                segment=index,
                kind=kind,
                start_state_boundary=start,
                stop_state_boundary=stop,
                start_control_inclusive=start,
                stop_control_exclusive=stop,
                controls=stop - start,
                start_time_seconds=start * dt,
                stop_time_seconds=stop * dt,
                duration_seconds=(stop - start) * dt,
                two_observed_onset_boundaries=kind == "complete_onset_interval",
                right_onset_contact_row=None if index == 0 else start - 1,
            )
        )
    return rows


def frame_schedule(
    controls: int, *, dt: float = DT, every: int = 4, terminal_hold_frames: int = 0
) -> list[dict]:
    """Map video PTS to measured state boundaries, including partial final timing."""
    if type(controls) is not int or not 0 <= controls <= 1000:
        raise ValueError("An actual prefix of at most1000 controls is required")
    if dt != DT or every != 4 or type(terminal_hold_frames) is not int:
        raise ValueError("J1 video timing is fixed at0.008s and every4 controls")
    if not 0 <= terminal_hold_frames <= 32:
        raise ValueError("Terminal display hold must be explicit and at most32 frames")
    indices = list(range(0, controls + 1, every))
    if indices[-1] != controls:
        indices.append(controls)
    result = []
    for state_index in indices:
        result.append(
            dict(
                frame=len(result),
                state_index=state_index,
                completed_controls=state_index,
                last_completed_control=None if state_index == 0 else state_index - 1,
                physical_time_seconds=state_index * dt,
                presentation_time_seconds=state_index * dt,
                presentation_tick_125hz=state_index,
                kind="initial"
                if state_index == 0
                else "actual_terminal"
                if state_index == controls
                else "actual_sample",
                terminal_display_hold=False,
            )
        )
    for ordinal in range(1, terminal_hold_frames + 1):
        result.append(
            dict(
                frame=len(result),
                state_index=controls,
                completed_controls=controls,
                last_completed_control=None if controls == 0 else controls - 1,
                physical_time_seconds=controls * dt,
                presentation_time_seconds=(controls + ordinal * every) * dt,
                presentation_tick_125hz=controls + ordinal * every,
                kind="terminal_display_hold",
                terminal_display_hold=True,
            )
        )
    return result


def encoder_binding(path: str | Path | None = None, expected_sha256: str | None = None) -> dict:
    """Discover once, then authenticate the exact encoder at render time."""
    executable = Path(path or shutil.which("ffmpeg") or "")
    if not str(executable) or not executable.is_file():
        raise FileNotFoundError("A preflight-pinned ffmpeg executable is required")
    executable = executable.resolve()
    digest = _sha(executable)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("Pinned ffmpeg bytes changed")
    result = subprocess.run(
        [str(executable), "-version"], capture_output=True, check=True, timeout=5
    )
    return dict(
        path=str(executable),
        bytes=executable.stat().st_size,
        sha256=digest,
        version=result.stdout.decode(errors="replace").splitlines()[0],
    )


def renderer_preflight(
    xml_path: str | Path,
    *,
    ffmpeg_path: str | Path | None = None,
    ffmpeg_sha256: str | None = None,
) -> dict:
    """Check offline native rendering; record an explicit schematic alternative."""
    import mujoco

    path = Path(xml_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    data = mujoco.MjData(model)
    available, reason, renderer = False, None, None
    counters = dict(offline_mj_forward_calls=0, offline_position_forward_calls=0)
    try:
        if os.environ.get("MUJOCO_GL") == "disable":
            reason = "MUJOCO_GL=disable; native GL context unavailable by configuration"
        else:
            try:
                mujoco.mj_forward(model, data)
                counters["offline_mj_forward_calls"] += 1
                renderer = mujoco.Renderer(model, height=240, width=320)
                renderer.update_scene(data)
                pixels = renderer.render()
                if pixels.shape != (240, 320, 3) or pixels.dtype != np.uint8:
                    raise ValueError("Native renderer returned an unexpected image")
                available = True
            except Exception as error:
                reason = f"{type(error).__name__}: {error}"
        if not available:
            mujoco.mj_fwdPosition(model, data)
            counters["offline_position_forward_calls"] += 1
            if not np.isfinite(data.geom_xpos).all():
                raise ValueError("Offline schematic geometry is unavailable")
    finally:
        if renderer is not None:
            renderer.close()
    encoder = encoder_binding(ffmpeg_path, ffmpeg_sha256)
    return dict(
        schema="walker-j1-atlas-render-preflight-v1",
        xml=dict(path=str(path), bytes=path.stat().st_size, sha256=_sha(path)),
        encoder=encoder,
        encoder_functional_probe=encoder_functional_probe(encoder),
        native_available=available,
        native_unavailable_reason=reason,
        mode="mujoco_snapshot" if available else "kinematic_schematic",
        schematic_rule="Actual XML geom positions/orientations after offline mj_fwdPosition; not a native camera image",
        work=dict(**counters, native_controls=0, physics_substeps=0, neural_rows=0),
    )


class _Budget:
    def __init__(self, out: Path, wall: float, maximum: int, rss: int):
        if not 0 < wall <= MAX_WALL or not 0 < maximum <= MAX_BYTES or not 0 < rss <= MAX_RSS:
            raise ValueError("Atlas budget exceeds the declared J1 allocation")
        self.out, self.started, self.deadline = out, time.monotonic(), time.monotonic() + wall
        self.maximum, self.rss_limit, self.files = maximum, rss, {}
        self.peak_rss = 0

    def check(self, current: Path | None = None) -> None:
        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        self.peak_rss = max(self.peak_rss, int(usage if sys.platform == "darwin" else usage * 1024))
        if current is not None and current.exists():
            self.files[current] = current.stat().st_size
        if time.monotonic() >= self.deadline or self.peak_rss > self.rss_limit:
            raise RuntimeError("Atlas wall/RSS reservation exhausted")
        if sum(self.files.values()) > self.maximum - 4 * 1024**2:
            raise RuntimeError("Atlas output reservation exhausted")

    def write(self, path: Path, payload: bytes) -> None:
        self.check()
        if sum(self.files.values()) + len(payload) > self.maximum - 4 * 1024**2:
            raise RuntimeError("Atlas output reservation exhausted before write")
        with path.open("xb") as stream:
            stream.write(payload)
        self.files[path] = len(payload)

    def json(self, path: Path, value: Any) -> None:
        self.write(path, (json.dumps(value, indent=2, allow_nan=False) + "\n").encode())

    def closing_json(self, path: Path, value: Any) -> None:
        """Use the shared emergency reserve after a rendering-budget stop."""
        payload = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        if len(payload) > 4 * 1024**2 or sum(self.files.values()) + len(payload) > self.maximum:
            raise RuntimeError("Atlas closing metadata exceeds emergency reservation")
        with path.open("xb") as stream:
            stream.write(payload)
        self.files[path] = len(payload)


def _trace(value: Mapping[str, Any]) -> dict:
    arrays = {name: np.asarray(value[name]) for name in ("states", "actions", "contacts", "forces")}
    count = len(arrays["actions"])
    expected = dict(states=(count + 1, 19), actions=(count, 6), contacts=(count, 2))
    for name, shape in expected.items():
        if arrays[name].shape != shape:
            raise ValueError("Atlas trace shape differs: " + name)
    if count > 1000 or arrays["contacts"].dtype.kind != "b":
        raise ValueError("Actual boolean contacts and at most1000 controls are required")
    if arrays["forces"].shape[:2] != (count, 2) or arrays["forces"].ndim not in {2, 3}:
        raise ValueError("Recorded forces need two foot channels per actual control")
    if arrays["forces"].ndim == 3 and arrays["forces"].shape[2] not in {1, 3, 6}:
        raise ValueError("Empty or unidentified force components are missing evidence")
    if not all(np.isfinite(array).all() for array in arrays.values()):
        raise ValueError("Nonfinite actual atlas measurements")
    times = arrays["states"][:, 0]
    if not np.allclose(np.diff(times), DT, rtol=0, atol=1e-10):
        raise ValueError("Atlas state timestamps do not match actual control timing")
    done = np.asarray(value.get("done", False))
    if done.dtype.kind != "b":
        raise ValueError("Actual done flags must be boolean")
    if done.shape == ():
        done = (
            np.r_[np.zeros(max(0, count - 1), dtype=bool), bool(done)]
            if count
            else np.zeros(0, bool)
        )
    if done.shape != (count,) or np.any(done[:-1]):
        raise ValueError("Atlas refuses fabricated controls after an actual fall")
    arrays["done"] = done.astype(bool)
    for name in ("observations", "warmstarts", "integrations", "costs", "fall_costs", "velocities"):
        if name in value:
            array = np.asarray(value[name])
            if array.dtype.kind not in "biuf" or not np.isfinite(array).all():
                raise ValueError("Invalid optional atlas array: " + name)
            arrays[name] = array
    return arrays


class _OfflineFrames:
    def __init__(self, xml: Path, mode: str):
        import mujoco

        self.mj, self.mode = mujoco, mode
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.renderer = (
            mujoco.Renderer(self.model, height=240, width=320)
            if mode == "mujoco_snapshot"
            else None
        )
        self.work = dict(
            offline_restores=0, offline_mj_forward_calls=0, offline_position_forward_calls=0
        )

    def frame(self, arrays: dict, index: int, tag: str) -> np.ndarray:
        from PIL import Image, ImageDraw

        mj, model, data = self.mj, self.model, self.data
        if "integrations" not in arrays:
            raise ValueError("Offline video requires the complete saved integration context")
        integration = arrays["integrations"]
        spec = mj.mjtState.mjSTATE_INTEGRATION
        if integration.shape != (len(arrays["states"]), mj.mj_stateSize(model, spec)):
            raise ValueError("Saved video integration dimension differs from pinned model")
        mj.mj_setState(model, data, integration[index], spec)
        restored = np.r_[data.time, data.qpos, data.qvel]
        if not np.array_equal(restored, arrays["states"][index]):
            raise ValueError("Video integration and plotted physical state disagree")
        if "warmstarts" in arrays:
            data.qacc_warmstart[:] = arrays["warmstarts"][index]
        self.work["offline_restores"] += 1
        if self.renderer is not None:
            mj.mj_forward(model, data)
            self.work["offline_mj_forward_calls"] += 1
            camera = mj.MjvCamera()
            camera.lookat[:] = [float(data.qpos[0]), 0, 0.8]
            camera.distance, camera.azimuth, camera.elevation = 3.5, 90, -15
            self.renderer.update_scene(data, camera=camera)
            image = Image.fromarray(self.renderer.render())
            label = "MuJoCo saved-snapshot render"
        else:
            mj.mj_fwdPosition(model, data)
            self.work["offline_position_forward_calls"] += 1
            image = Image.new("RGB", (320, 240), "#f5f5f2")
            draw = ImageDraw.Draw(image)
            root = float(data.qpos[0])

            def pixel(position):
                return int(160 + (position[0] - root) * 105), int(212 - position[2] * 105)

            draw.line([(0, 212), (320, 212)], fill="#777777", width=2)
            for geom in range(model.ngeom):
                if int(model.geom_type[geom]) != int(mj.mjtGeom.mjGEOM_CAPSULE):
                    continue
                center = data.geom_xpos[geom]
                axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
                extent = float(model.geom_size[geom, 1])
                color = tuple((np.clip(model.geom_rgba[geom, :3], 0, 1) * 255).astype(int))
                width = max(2, int(model.geom_size[geom, 0] * 210))
                draw.line(
                    [pixel(center - extent * axis), pixel(center + extent * axis)],
                    fill=color,
                    width=width,
                )
            label = "KINEMATIC SCHEMATIC - not camera image"
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 320, 35), fill="white")
        draw.text((3, 2), label, fill="black")
        draw.text((3, 17), f"{tag[:29]} state={index} t={index * DT:.3f}s", fill="black")
        return np.asarray(image, dtype=np.uint8)

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()


def _video_command(executable: str, path: Path, count: int) -> list[str]:
    regular = len(range(0, count + 1, 4))
    extra = int(count % 4 != 0)
    pts = f"if(lt(N,{regular}),4*N,{count}+4*(N-{regular}+{1 - extra}))"
    return [
        executable,
        "-n",
        "-loglevel",
        "error",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        "320x240",
        "-r",
        "31.25",
        "-i",
        "pipe:0",
        "-an",
        "-vf",
        "settb=expr=1/125,setpts='" + pts + "'",
        "-fps_mode",
        "vfr",
        "-enc_time_base",
        "1:125",
        "-video_track_timescale",
        "125",
        "-threads",
        "1",
        "-filter_threads",
        "1",
        "-vcodec",
        "libx264",
        "-bf",
        "0",
        "-preset",
        "fast",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]


def encoder_functional_probe(encoder: Mapping[str, Any]) -> dict:
    """Encode three calibration frames with the exact final video command."""
    started = time.monotonic()
    deadline = started + 20
    sibling = Path(encoder["path"]).parent / "ffprobe"
    if not sibling.is_file():
        raise FileNotFoundError("The pinned ffmpeg family must include sibling ffprobe")
    ffprobe = encoder_binding(sibling)

    def remaining(maximum: float) -> float:
        value = min(maximum, deadline - time.monotonic())
        if value <= 0:
            raise RuntimeError("Encoder functional preflight wall budget exhausted")
        return value

    with tempfile.TemporaryDirectory(prefix="walker-j1-encoder-probe-") as directory:
        root = Path(directory)
        video, errors = root / "calibration.mp4", root / "encoder_stderr.txt"
        frames = np.zeros((3, 240, 320, 3), dtype=np.uint8)
        frames[1] = 64
        frames[2] = 128
        command = _video_command(str(encoder["path"]), video, 7)
        with errors.open("xb") as error_stream:
            result = subprocess.run(
                command,
                input=frames.tobytes(),
                stdout=subprocess.DEVNULL,
                stderr=error_stream,
                timeout=remaining(15),
                check=True,
            )
        probe = subprocess.run(
            [
                ffprobe["path"],
                "-v",
                "error",
                "-show_entries",
                "frame=pts_time",
                "-of",
                "json",
                str(video),
            ],
            capture_output=True,
            timeout=remaining(5),
            check=True,
        )
        times = [float(row["pts_time"]) for row in json.loads(probe.stdout)["frames"]]
        if len(times) != 3 or not np.allclose(times, [0, 0.032, 0.056], rtol=0, atol=1e-9):
            raise ValueError("Pinned encoder does not preserve J1 partial-terminal VFR timestamps")
        size = sum(path.stat().st_size for path in root.iterdir() if path.is_file())
        if size > 2 * 1024**2:
            raise RuntimeError("Encoder functional preflight temporary output exceeds two MiB")
        return dict(
            schema="walker-j1-encoder-functional-probe-v1",
            passed=True,
            command=command,
            encoder_exit_code=result.returncode,
            ffprobe=ffprobe,
            observed_presentation_times_seconds=times,
            expected_presentation_times_seconds=[0, 0.032, 0.056],
            calibration_frames=3,
            synthetic_control_boundary_labels=[0, 4, 7],
            actual_scientific_controls=0,
            native_controls=0,
            physics_substeps=0,
            neural_rows=0,
            fitting_updates=0,
            temporary_output_bytes=size,
            temporary_output_limit_bytes=2 * 1024**2,
            temporary_output_removed=True,
            wall_seconds=time.monotonic() - started,
            wall_limit_seconds=20,
        )


def _video(
    arrays: dict,
    tag: str,
    out: Path,
    schedule: list[dict],
    backend: Any,
    encoder: dict,
    budget: _Budget,
) -> dict:
    path, errors = out / "video.mp4", out / "encoder_stderr.txt"
    command = _video_command(encoder["path"], path, len(arrays["actions"]))
    started, frames, last = time.monotonic(), 0, None
    receipt = dict(
        command=command, rendering_only=True, frames=0, native_controls=0, physics_substeps=0
    )
    with errors.open("xb") as error_stream:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=error_stream,
            start_new_session=True,
        )
        receipt["pid"] = process.pid
        assert process.stdin is not None
        os.set_blocking(process.stdin.fileno(), False)
        selector = selectors.DefaultSelector()
        selector.register(process.stdin, selectors.EVENT_WRITE)
        try:
            for row in schedule:
                budget.check(path)
                budget.check(errors)
                if not row["terminal_display_hold"]:
                    last = backend.frame(arrays, row["state_index"], tag)
                if last is None or last.shape != (240, 320, 3) or last.dtype != np.uint8:
                    raise ValueError("Offline video frame does not match320x240RGB")
                pending = memoryview(last).cast("B")
                while pending:
                    budget.check(path)
                    budget.check(errors)
                    if process.poll() is not None:
                        raise RuntimeError("Video encoder exited before receiving all frames")
                    if selector.select(0.1):
                        try:
                            sent = os.write(process.stdin.fileno(), pending)
                            pending = pending[sent:]
                        except BlockingIOError:
                            pass
                frames += 1
            process.stdin.close()
            while process.poll() is None:
                budget.check(path)
                budget.check(errors)
                time.sleep(min(0.02, max(0.001, budget.deadline - time.monotonic())))
            if process.wait() != 0:
                raise RuntimeError("Video encoder failed; see preserved encoder_stderr.txt")
            receipt.update(passed=True, exit_code=process.returncode, leader_reaped=True)
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            receipt.update(
                passed=False, exit_code=process.returncode, leader_reaped=True, error=str(error)
            )
            raise
        finally:
            selector.close()
            if not process.stdin.closed:
                process.stdin.close()
            budget.files[errors] = errors.stat().st_size
            if path.exists():
                budget.files[path] = path.stat().st_size
            receipt.update(frames=frames, wall_seconds=time.monotonic() - started)
            budget.closing_json(out / "encoder_receipt.json", receipt)
    return receipt


def _plots(arrays: dict, segments: list[dict], tag: str) -> bytes:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    states, actions, contacts = (arrays[key] for key in ("states", "actions", "contacts"))
    count = len(actions)
    state_time = states[:, 0] - states[0, 0]
    after = state_time[1:]
    components = int(np.prod(arrays["forces"].shape[2:]))
    force = arrays["forces"].reshape(count, 2, components)
    norms = np.linalg.norm(force[:, :, :3], axis=2) if components > 1 else force[:, :, 0]
    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    for index, label in enumerate(
        ["pitch", "right hip", "right knee", "right ankle", "left hip", "left knee", "left ankle"]
    ):
        axes[0].plot(state_time, states[:, 3 + index], linewidth=0.8, label=label)
    axes[0].legend(ncol=4, fontsize=7)
    for foot, name in enumerate(["right", "left"]):
        axes[1].step(after, contacts[:, foot].astype(int) + 1.3 * foot, where="post", label=name)
        axes[2].plot(after, norms[:, foot], label=name, linewidth=0.8)
    for index in range(6):
        axes[3].plot(state_time[:-1], actions[:, index], label=str(index), linewidth=0.8)
    for axis, label in zip(
        axes,
        [
            "angle (rad)",
            "recorded contact (+offset)",
            "recorded force magnitude",
            "applied control",
        ],
        strict=True,
    ):
        axis.set_ylabel(label)
        for segment in segments:
            if segment["right_onset_contact_row"] is not None:
                axis.axvline(segment["start_time_seconds"], color="gray", alpha=0.18, linewidth=0.5)
        axis.grid(alpha=0.15)
    axes[-1].set_xlabel("Actual seconds from initial snapshot; no post-fall samples")
    fig.suptitle(tag + ": all actual samples; raw right-contact onset boundaries")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=110)
    plt.close(fig)
    return buffer.getvalue()


def _csv(rows: list[dict]) -> bytes:
    if not rows:
        return b""
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode()


def render_atlas(
    traces: Iterable[Mapping[str, Any]] | Mapping[str, Mapping[str, Any]],
    out: str | Path,
    *,
    xml_path: str | Path,
    mode: str,
    ffmpeg_path: str | Path,
    ffmpeg_sha256: str,
    xml_sha256: str | None = None,
    terminal_hold_frames: int = 0,
    wall_seconds: float = MAX_WALL,
    output_bytes: int = MAX_BYTES,
    peak_rss_bytes: int = MAX_RSS,
) -> dict:
    """Render the predeclared iterator into a create-only, fully mapped atlas."""
    if mode not in {"mujoco_snapshot", "kinematic_schematic"}:
        raise ValueError("An explicitly preflight-pinned rendering mode is required")
    target, xml = Path(out), Path(xml_path).resolve()
    if target.exists():
        raise FileExistsError("Atlas output is create-only")
    if xml_sha256 is not None and _sha(xml) != xml_sha256:
        raise ValueError("Pinned source geometry XML changed")
    encoder = encoder_binding(ffmpeg_path, ffmpeg_sha256)
    target.mkdir(parents=True, exist_ok=False)
    budget = _Budget(target, wall_seconds, output_bytes, peak_rss_bytes)
    records, total_frames, backend = [], 0, None
    work = dict(offline_restores=0, offline_mj_forward_calls=0, offline_position_forward_calls=0)
    receipt = dict(schema="walker-j1-stride-atlas-v1", passed=False, mode=mode, encoder=encoder)
    iterator = (
        (dict(value, tag=key) for key, value in traces.items())
        if isinstance(traces, Mapping)
        else iter(traces)
    )
    try:
        backend = _OfflineFrames(xml, mode)
        for index, trace in enumerate(iterator):
            budget.check()
            if index >= MAX_TRACES:
                raise RuntimeError("Atlas predeclared trace limit exceeded")
            tag = str(trace["tag"])
            if not 0 < len(tag) <= 256:
                raise ValueError("Atlas trace tag must contain at most256 characters")
            arrays = _trace(trace)
            count = len(arrays["actions"])
            schedule = frame_schedule(count, terminal_hold_frames=terminal_hold_frames)
            total_frames += len(schedule)
            if total_frames > MAX_FRAMES:
                raise RuntimeError("Atlas frame/restore/forward reservation exhausted")
            directory = target / f"trace_{index:03d}"
            directory.mkdir()
            segments = stride_segments(arrays["contacts"])
            budget.write(directory / "strides.csv", _csv(segments))
            for row in schedule:
                row["absolute_simulator_time_seconds"] = float(
                    arrays["states"][row["state_index"], 0]
                )
            budget.write(directory / "frames.csv", _csv(schedule))
            budget.write(directory / "motion_contacts_controls.png", _plots(arrays, segments, tag))
            scalar = {
                name: value
                for name, value in arrays.items()
                if name not in {"integrations", "warmstarts"}
            }
            buffer = io.BytesIO()
            np.savez_compressed(buffer, **scalar)
            budget.write(directory / "actual_scalars.npz", buffer.getvalue())
            video = _video(arrays, tag, directory, schedule, backend, encoder, budget)
            record = dict(
                tag=tag,
                directory=directory.name,
                actual_controls=count,
                actual_states=count + 1,
                fell=bool(count and arrays["done"][-1]),
                actual_duration_seconds=count * DT,
                segments=segments,
                observed_right_onsets=sum(
                    row["right_onset_contact_row"] is not None for row in segments
                ),
                input_arrays={name: _binding(value) for name, value in arrays.items()},
                video=video,
                video_frame_map="frames.csv",
                contact_source=str(
                    trace.get(
                        "contact_source",
                        "supplied actual post-control cache measurements; no forward-generated contacts substituted",
                    )
                ),
                geometry_source=dict(xml_sha256=_sha(xml), mode=mode),
            )
            budget.json(directory / "trace.json", record)
            records.append(
                {key: value for key, value in record.items() if key != "segments"}
                | dict(
                    stride_segments_file="strides.csv",
                    stride_segments_count=len(segments),
                )
            )
        work.update(backend.work)
        cards = []
        for row in records:
            directory, tag = row["directory"], html.escape(row["tag"])
            cards.append(
                f"<section><h2>{tag}</h2><p>{row['actual_controls']} actual controls; "
                f"fall={row['fell']}; video mode={html.escape(mode)}</p>"
                f'<video controls preload="metadata" src="{directory}/video.mp4"></video>'
                f'<img alt="All actual samples" src="{directory}/motion_contacts_controls.png">'
                f'<p><a href="{directory}/strides.csv">Raw stride intervals</a> · '
                f'<a href="{directory}/frames.csv">Frame/control/time map</a> · '
                f'<a href="{directory}/actual_scalars.npz">All actual scalar arrays</a> · '
                f'<a href="{directory}/trace.json">Bindings and video receipt</a></p></section>'
            )
        page = (
            '<!doctype html><meta charset="utf-8"><title>J1 stride atlas</title>'
            "<style>body{font:16px system-ui;max-width:1100px;margin:32px auto}img{width:100%}"
            "video{width:640px;max-width:100%}section{border-top:1px solid #ccc;margin-top:30px}</style>"
            "<h1>J1: actual joint motion, measured contacts and applied controls</h1>"
            "<p>Raw right-foot false→true onsets, with no debounce, phase warp or periodicity assumption. "
            "Partial intervals, spikes, dropouts and actual falls remain present. "
            "These event partitions are not certified periodic biomechanical strides. "
            "Contact rows are measured after control j, at state boundary j+1. "
            "Frames use measured state boundaries at0,4,…,N and "
            "125Hz timestamp units; the partial terminal interval is not stretched. Final coded endpoint "
            "display duration is a video presentation hold, not a new physical observation. Any added holds "
            "are separately labelled in every frame map. No post-fall state is fabricated.</p>"
            f"<p>Rendering mode: <strong>{html.escape(mode)}</strong>. "
            "Kinematic schematics use source XML geometry and offline position forwards; they are not native "
            "camera images. Neither rendering mode supplies scientific contact measurements.</p>"
            + "".join(cards)
        )
        budget.write(target / "index.html", page.encode())
        receipt.update(passed=True)
    except BaseException as error:
        receipt.update(error=dict(type=type(error).__name__, message=str(error)))
        raise
    finally:
        if backend is not None:
            work.update(backend.work)
            backend.close()
        receipt.update(
            traces=records,
            frames=total_frames,
            work=dict(
                **work, native_controls=0, physics_substeps=0, neural_rows=0, optimizer_steps=0
            ),
            wall_seconds=time.monotonic() - budget.started,
            peak_rss_bytes=budget.peak_rss,
            output_bytes_before_receipt=sum(budget.files.values()),
            limits=dict(
                wall_seconds=wall_seconds,
                output_bytes=output_bytes,
                peak_rss_bytes=peak_rss_bytes,
                frames=MAX_FRAMES,
                traces=MAX_TRACES,
            ),
            timing=dict(
                control_dt=DT,
                nominal_video_fps=31.25,
                timestamp_units_per_second=125,
                sampling_every_controls=4,
                explicit_terminal_hold_frames=terminal_hold_frames,
            ),
        )
        budget.closing_json(target / "atlas_receipt.json", receipt)
    return receipt
