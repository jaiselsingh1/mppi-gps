"""Offline schematic playback of recorded Acrobot episode boundaries.

No environment, engine, actor, trainer or planner is imported or executed.
Declared episode semantics and a file digest bind the input; this visualization
does not independently certify the controller that generated those records.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

import numpy as np


_SEMANTICS = frozenset(("actor", "executed_replanning", "analytic_reference"))


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stat(path):
    value = Path(path).stat()
    return dict(size=value.st_size, mtime_ns=value.st_mtime_ns,
                device=value.st_dev, inode=value.st_ino)


def _write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def _load_episode(path):
    with np.load(path, allow_pickle=False) as raw:
        required = ("states", "warmstarts", "elapsed", "hold_counts", "observations",
            "actions", "stage_costs", "terminal_costs", "capture_failures", "component_costs",
            "tip_distances", "query_ids", "execution_semantics", "actor_owner_json",
            "source_json", "initial_snapshot_json", "complete", "captured", "T", "family")
        if not set(required) <= set(raw.files):
            raise ValueError("Input must be an archived actual EpisodeTrace NPZ")
        arrays = {name: raw[name].copy() for name in required}
    semantics = str(arrays["execution_semantics"])
    if semantics not in _SEMANTICS:
        raise ValueError("Only actual actor, executed-replanning or analytic-reference traces may be rendered")
    states, actions = arrays["states"], arrays["actions"]
    if actions.ndim != 2 or actions.shape[1] != 1 or len(actions) < 1:
        raise ValueError("Actual actions must have shape (N,1), N>0")
    n = len(actions)
    expected = dict(states=(n+1,5), warmstarts=(n+1,2), elapsed=(n+1,),
        hold_counts=(n+1,), observations=(n+1,8), actions=(n,1), stage_costs=(n,),
        terminal_costs=(n,), capture_failures=(n,), component_costs=(n,4), tip_distances=(n+1,))
    for name, shape in expected.items():
        values = arrays[name]
        if values.shape != shape or values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError(f"Invalid finite aligned actual array: {name}")
    if np.any(abs(actions) > 1.):
        raise ValueError("Saved actions exceed native Acrobot control bounds")
    elapsed = arrays["elapsed"]
    if (not np.array_equal(elapsed, np.round(elapsed)) or elapsed[0] < 0
            or not np.array_equal(np.diff(elapsed), np.ones(n))):
        raise ValueError("Saved episode clock must advance one actual control per row")
    if not np.allclose(states[:,0], elapsed*.01, rtol=0, atol=1e-9):
        raise ValueError("Saved physics time differs from the .01-second explicit control clock")
    hold = arrays["hold_counts"]
    if not np.array_equal(hold, np.round(hold)) or np.any(hold < 0) or np.any(hold > elapsed):
        raise ValueError("Invalid saved capture-history counts")
    ids = arrays["query_ids"]
    if (ids.shape != (n,) or ids.dtype.kind not in "US"
            or len(set(ids.tolist())) != n or not all(str(item) for item in ids)):
        raise ValueError("Every actual control must retain its unique raw query ID")
    T = int(arrays["T"])
    if T < elapsed[-1] or (bool(arrays["complete"]) and (elapsed[0] != 0 or elapsed[-1] != T)):
        raise ValueError("Saved completion flag and finite horizon disagree")
    metadata = {}
    for name in ("actor_owner_json", "source_json", "initial_snapshot_json"):
        metadata[name] = json.loads(str(arrays[name]))
        json.dumps(metadata[name], allow_nan=False)
    initial = metadata["initial_snapshot_json"]
    if (not np.array_equal(initial["state"], states[0])
            or not np.array_equal(initial["warmstart"], arrays["warmstarts"][0])
            or initial["elapsed"] != elapsed[0] or initial["hold_count"] != hold[0]):
        raise ValueError("Initial snapshot does not match the first recorded boundary")
    return arrays, metadata


def render_episode(episode_path, out_dir, expected_sha256=None, fps=30,
                   format="mp4", max_frames=1200):
    """Stream saved-state schematic frames to one explicitly chosen encoder.

    Frames choose the nearest recorded boundary to each real-time display
    sample. No interpolated or simulated states are introduced. The true final
    boundary is always included. Exceeding the declared frame cap fails before
    encoding, rather than silently truncating the trajectory. No format fallback
    or retry occurs. A failure preserves the partial file and a failure receipt.
    """
    started = time.perf_counter()
    episode = Path(episode_path).resolve()
    out = Path(out_dir)
    out.mkdir(parents=True, mode=0o700, exist_ok=False)
    common = dict(schema="bidirectional-saved-state-render-v1",
        started_utc=datetime.now(timezone.utc).isoformat(), episode=str(episode),
        expected_sha256=expected_sha256, format=str(format), fps=fps, max_frames=max_frames,
        renderer_source_sha256=_sha256(__file__),
        new_simulator_controls=0, new_physics_frames=0, new_actor_inference_rows=0,
        new_fits=0, scope="Offline saved-state schematic; no engine or controller execution")
    _write_json(out/"started.json", common)
    output = None
    appended = 0
    try:
        if format not in ("mp4", "gif"):
            raise ValueError("Explicit format must be mp4 or gif")
        if type(fps) not in (int,float) or not np.isfinite(fps) or fps <= 0:
            raise ValueError("Positive finite FPS required")
        if type(max_frames) is not int or max_frames < 2:
            raise ValueError("max_frames must be an integer >=2")
        before = _stat(episode)
        digest = _sha256(episode)
        if expected_sha256 is not None and digest != str(expected_sha256).lower():
            raise ValueError("Episode digest differs from expected_sha256")
        if _stat(episode) != before:
            raise RuntimeError("Episode changed while its initial digest was read")
        arrays, metadata = _load_episode(episode)
        if _stat(episode) != before or _sha256(episode) != digest:
            raise RuntimeError("Episode changed while its arrays were loaded")
        states = arrays["states"]
        times = states[:,0]-states[0,0]
        duration = float(times[-1])
        display_times = np.arange(int(np.floor(duration*fps))+1, dtype=float)/fps
        if abs(display_times[-1]-duration) > 1e-12:
            display_times = np.append(display_times, duration)
        else:
            display_times[-1] = duration
        if len(display_times) > max_frames:
            raise ValueError(f"Full trajectory needs {len(display_times)} frames, exceeding cap {max_frames}")
        right = np.searchsorted(times, display_times, side="left").clip(0,len(times)-1)
        left = np.maximum(right-1,0)
        indices = np.where(abs(times[left]-display_times) <= abs(times[right]-display_times), left, right)
        indices[0], indices[-1] = 0, len(states)-1
        output = out/("episode."+format)
        if output.exists():
            raise FileExistsError(output)
        # Lazy graphics/encoder loading. A private temporary font/config cache
        # avoids incidental user-directory writes; neither import is an engine.
        temporary_root = "/private/tmp" if Path("/private/tmp").is_dir() else "/tmp"
        with tempfile.TemporaryDirectory(prefix="mppi-gps-fontcache-",dir=temporary_root) as cache:
            previous_cache = os.environ.get("MPLCONFIGDIR")
            os.environ["MPLCONFIGDIR"] = cache
            try:
                import matplotlib
                matplotlib.use("Agg", force=True)
                from matplotlib.figure import Figure
                from matplotlib.backends.backend_agg import FigureCanvasAgg
                from matplotlib.patches import Circle
                import imageio.v2 as imageio
                figure = Figure(figsize=(6,6),dpi=120)
                canvas = FigureCanvasAgg(figure)
                ax = figure.add_subplot(111)
                ax.set(xlim=(-2.35,2.35),ylim=(-.25,4.55),aspect="equal",
                       xlabel="horizontal position (m)",ylabel="height (m)")
                ax.axhline(0,color=".75",linewidth=1)
                ax.add_patch(Circle((0,4),.2,fill=False,color="green",linewidth=1.5))
                ax.plot(0,2,"ko",markersize=7)
                upper, = ax.plot([],[],"o-",color="#2474b5",linewidth=5,markersize=7)
                lower, = ax.plot([],[],"o-",color="#e98128",linewidth=5,markersize=7)
                title = ax.set_title("")
                footer = figure.text(.5,.015,"Recorded state · schematic playback · no simulator/policy rerun",
                                     ha="center",fontsize=8)
                figure.tight_layout(rect=(0,.03,1,.96))
                cost = np.r_[0.,np.cumsum(arrays["stage_costs"]+arrays["terminal_costs"]+arrays["capture_failures"])]
                options = dict(format="FFMPEG",mode="I",fps=fps,codec="libx264",macro_block_size=1) if format=="mp4" else dict(
                    format="GIF",mode="I",duration=1000./fps,loop=0)
                with imageio.get_writer(str(output),**options) as writer:
                    for frame_index, boundary in enumerate(indices):
                        q0,q1 = states[boundary,1:3]
                        elbow = (np.sin(q0),2.+np.cos(q0))
                        tip = (elbow[0]+np.sin(q0+q1),elbow[1]+np.cos(q0+q1))
                        upper.set_data([0.,elbow[0]],[2.,elbow[1]])
                        lower.set_data([elbow[0],tip[0]],[elbow[1],tip[1]])
                        title.set_text(f"Saved Acrobot trajectory ({str(arrays['execution_semantics'])})\n"
                            f"t={states[boundary,0]:.2f} s · control {int(arrays['elapsed'][boundary])} · cost {cost[boundary]:.3f}")
                        canvas.draw()
                        frame = np.asarray(canvas.buffer_rgba())[...,:3].copy()
                        writer.append_data(frame)
                        appended += 1
                figure.clear()
            finally:
                if previous_cache is None:
                    os.environ.pop("MPLCONFIGDIR",None)
                else:
                    os.environ["MPLCONFIGDIR"] = previous_cache
        after, final_digest = _stat(episode), _sha256(episode)
        if after != before or final_digest != digest:
            raise RuntimeError("Source episode changed during rendering")
        if _sha256(__file__) != common["renderer_source_sha256"]:
            raise RuntimeError("Renderer source changed during execution")
        result = dict(**common,status="COMPLETED",finished_utc=datetime.now(timezone.utc).isoformat(),
            source_sha256=digest,source_stat_before=before,source_stat_after=after,
            execution_semantics=str(arrays["execution_semantics"]),
            actor_owner=metadata["actor_owner_json"],source_provenance=metadata["source_json"],
            context_id=metadata["initial_snapshot_json"]["context_id"],
            task_hash=metadata["initial_snapshot_json"]["task_hash"],
            frames=appended,recorded_boundaries=len(states),recorded_controls=len(arrays["actions"]),
            first_boundary=int(indices[0]),last_boundary=int(indices[-1]),
            sampled_boundary_indices=indices.tolist(),sampling="nearest actual saved boundary; no interpolation",
            max_sampling_time_error_seconds=float(abs(times[indices]-display_times).max()),
            physical_duration_seconds=duration,encoded_constant_fps_duration_seconds=appended/fps,
            endpoint_frame_included=True,endpoint_extra_display_duration_at_most_seconds=1./fps,
            output=str(output.name),output_sha256=_sha256(output),output_bytes=output.stat().st_size,
            wall_seconds=time.perf_counter()-started,
            scientific_verdict="None; playback does not independently verify controller ownership or outcomes")
        _write_json(out/"render.json",result)
        return result
    except BaseException as error:
        failure = dict(**common,status="FAILED_PRESERVED_NO_RETRY",
            error_type=type(error).__name__,error=str(error),frames_appended=appended,
            wall_seconds=time.perf_counter()-started,
            partial_output=None if output is None else str(output.name))
        if output is not None and output.exists():
            failure.update(partial_output_bytes=output.stat().st_size,
                           partial_output_sha256=_sha256(output))
        _write_json(out/"render.json",failure)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode",required=True,type=Path)
    parser.add_argument("--out",required=True,type=Path)
    parser.add_argument("--expected-sha256")
    parser.add_argument("--fps",type=float,default=30.)
    parser.add_argument("--format",choices=("mp4","gif"),default="mp4")
    parser.add_argument("--max-frames",type=int,default=1200)
    args=parser.parse_args()
    result=render_episode(args.episode,args.out,args.expected_sha256,args.fps,args.format,args.max_frames)
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=="__main__":
    main()
