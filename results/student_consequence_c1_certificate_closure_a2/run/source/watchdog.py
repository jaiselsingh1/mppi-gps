from __future__ import annotations
import json,math,os,secrets,selectors,shutil,signal,subprocess,sys,time
from pathlib import Path
from typing import Any

def _rss_bytes(pid: int) -> int:
    # Linux removes VmRSS when a task has released its address space but its
    # zombie/dead status still exists. Re-read once for the exit race; absence
    # of RSS in a live task remains a watchdog failure, never a zero sample.
    for _ in range(2):
        try:
            lines = Path(f"/proc/{pid}/status").read_text().splitlines()
        except (FileNotFoundError, ProcessLookupError):
            return 0
        state = None
        for line in lines:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
            if line.startswith("State:"):
                fields = line.split()
                state = fields[1] if len(fields) > 1 else None
        if state in {"Z", "X", "x"}:
            return 0
    raise RuntimeError("Pinned Linux worker RSS could not be measured")


def bounded_process(
    command: list[str],
    *,
    cwd: Path,
    environment: dict[str, str],
    output: Path,
    wall_seconds: float,
    stdout_bytes: int,
    peak_rss_bytes: int | None = None,
) -> dict[str, Any]:
    """Kill/reap the entire worker group at a frozen outer limit; never retry."""
    if not command or any(not isinstance(arg, str) or "\x00" in arg for arg in command):
        raise ValueError("An explicit process command is required")
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds <= 0
        or type(stdout_bytes) is not int
        or stdout_bytes < 1
    ):
        raise ValueError("Positive finite wall and stdout reservations are required")
    if peak_rss_bytes is not None and (type(peak_rss_bytes) is not int or peak_rss_bytes < 1):
        raise ValueError("An explicit positive RSS reservation is required")
    started = time.monotonic()
    with output.open("xb") as stream:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        reason = None
        observed_rss = 0
        rss_exit_observations = 0
        selector = None
        try:
            selector = selectors.DefaultSelector()
            assert process.stdout is not None
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                # Poll even while stdout is registered. A dead leader can
                # retain unread output, but no longer has RSS to measure.
                return_code = process.poll()
                if return_code is not None and not selector.get_map():
                    break
                if time.monotonic() - started >= wall_seconds:
                    reason = "wall_limit"
                    break
                if peak_rss_bytes is not None and return_code is None:
                    try:
                        observed_rss = max(observed_rss, _rss_bytes(process.pid))
                    except Exception as rss_error:
                        # Linux can clear mm before publishing a terminal task
                        # state. Missing RSS alone never establishes exit: wait
                        # briefly for the actual child status, within the same
                        # deadline. A still-live child remains a hard failure.
                        if process.poll() is None:
                            remaining = wall_seconds - (time.monotonic() - started)
                            if remaining <= 0:
                                reason = "wall_limit"
                                break
                            try:
                                process.wait(timeout=min(0.05, remaining))
                            except subprocess.TimeoutExpired:
                                raise rss_error
                        rss_exit_observations += 1
                    if observed_rss > peak_rss_bytes:
                        reason = "rss_limit"
                        break
                for key, _ in selector.select(min(0.1, max(wall_seconds / 10, 0.001))):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    remaining = stdout_bytes - stream.tell()
                    stream.write(data[:remaining])
                    if len(data) > remaining:
                        reason = "stdout_storage_limit"
                        break
                if reason is not None:
                    break
            if reason is not None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        except BaseException as error:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            error.process_receipt = dict(
                command=command,
                pid=process.pid,
                exit_code=process.returncode,
                stopping_reason="coordinator_exception",
                leader_reaped=True,
                process_group_kill_requested=True,
                retries=0,
                wall_seconds=time.monotonic() - started,
                error=dict(type=type(error).__name__, message=str(error)),
                unknown_partial_work_possible=True,
                rss_unavailable_confirmed_exit_observations=rss_exit_observations,
            )
            raise
        finally:
            if selector is not None:
                selector.close()
            if process.stdout is not None:
                process.stdout.close()
        exit_code = process.wait()
    return dict(
        command=command,
        pid=process.pid,
        exit_code=exit_code,
        stopping_reason=reason,
        wall_seconds=time.monotonic() - started,
        stdout_bytes=output.stat().st_size,
        leader_reaped=True,
        process_group_kill_requested=reason is not None,
        retries=0,
        unknown_partial_work_possible=reason is not None or exit_code != 0,
        observed_peak_rss_bytes=observed_rss if peak_rss_bytes is not None else None,
        rss_unavailable_confirmed_exit_observations=rss_exit_observations,
    )

