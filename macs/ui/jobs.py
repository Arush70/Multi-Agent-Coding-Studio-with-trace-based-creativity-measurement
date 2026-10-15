"""Background jobs for the dashboard.

Building references and running the experiment take hours, so the dashboard starts the existing
scripts as separate processes instead of running them inside the page. A job keeps running if the
browser tab is refreshed or closed, and the page reads its progress from a log file.

Each job has a kind (one job of a kind at a time) and two files in data/jobs/:
  <kind>.log   everything the script prints
  <kind>.json  command, process id, start time, and, once finished, the exit code and end time

The job is launched through ``python -m macs.ui.jobs <kind>``: a small wrapper that runs the
script, then writes the exit code into <kind>.json, so a finished job is recognised even after
the dashboard itself has been restarted.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOBS_DIR = ROOT / "data" / "jobs"
MODEL_JOBS = {"references", "multirun", "smoke"}     # these call the model server

_PROGRESS = re.compile(r"\[(\d+)/(\d+)\]")


def _paths(kind: str, jobs_dir: Path = JOBS_DIR) -> tuple[Path, Path]:
    return jobs_dir / f"{kind}.json", jobs_dir / f"{kind}.log"


def _read_meta(kind: str, jobs_dir: Path = JOBS_DIR) -> dict | None:
    meta, _ = _paths(kind, jobs_dir)
    try:
        return json.loads(meta.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _write_meta(kind: str, data: dict, jobs_dir: Path = JOBS_DIR) -> None:
    meta, _ = _paths(kind, jobs_dir)
    meta.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta.with_name(meta.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=1), encoding="utf-8")
    os.replace(tmp, meta)


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if os.name == "nt":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"], capture_output=True, text=True)
        return str(pid) in out.stdout
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def status(kind: str, jobs_dir: Path = JOBS_DIR, tail: int = 40) -> dict:
    """{'state': 'none'|'running'|'finished'|'failed'|'stopped', 'returncode', 'progress', 'log', ...}"""
    meta = _read_meta(kind, jobs_dir)
    _, log = _paths(kind, jobs_dir)
    text = log.read_text(encoding="utf-8", errors="replace") if log.exists() else ""
    lines = text.splitlines()
    out = {"state": "none", "returncode": None, "progress": parse_progress(text),
           "log": "\n".join(lines[-tail:]), "meta": meta or {}}
    if not meta:
        return out
    if meta.get("stopped"):
        out["state"] = "stopped"
    elif meta.get("returncode") is not None:
        out["state"] = "finished" if meta["returncode"] == 0 else "failed"
        out["returncode"] = meta["returncode"]
    elif _pid_alive(int(meta.get("pid", -1))):
        out["state"] = "running"
    else:
        out["state"] = "failed"                     # wrapper vanished without writing an exit code
    return out


def parse_progress(text: str) -> tuple[int, int] | None:
    """The last '[k/N]' counter printed by a script, e.g. run_experiment.py's session counter."""
    found = _PROGRESS.findall(text or "")
    return (int(found[-1][0]), int(found[-1][1])) if found else None


def running_model_job(jobs_dir: Path = JOBS_DIR) -> str | None:
    for kind in MODEL_JOBS:
        if status(kind, jobs_dir)["state"] == "running":
            return kind
    return None


def start(kind: str, script: str, args: list[str], jobs_dir: Path = JOBS_DIR) -> dict:
    """Start scripts/<script> with args as job `kind`. Refuses if that kind is already running,
    or if another model job is running (they would compete for the same model server)."""
    if status(kind, jobs_dir)["state"] == "running":
        raise RuntimeError(f"a {kind} job is already running")
    if kind in MODEL_JOBS:
        other = running_model_job(jobs_dir)
        if other:
            raise RuntimeError(f"the {other} job is using the model; wait for it or stop it first")
    jobs_dir.mkdir(parents=True, exist_ok=True)
    meta_path, log_path = _paths(kind, jobs_dir)
    path = Path(script) if Path(script).is_absolute() else ROOT / "scripts" / script
    cmd = [sys.executable, "-u", str(path), *args]
    log_path.write_text("$ " + " ".join(cmd[2:]) + "\n", encoding="utf-8")
    _write_meta(kind, {"kind": kind, "cmd": cmd, "pid": -1, "started": time.strftime("%Y-%m-%d %H:%M:%S")},
                jobs_dir)
    wrapper = [sys.executable, "-m", "macs.ui.jobs", kind, str(jobs_dir)]
    env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
    kw: dict = {"cwd": str(ROOT), "env": env, "stdin": subprocess.DEVNULL,
                "stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        kw["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(subprocess, "CREATE_NO_WINDOW", 0)
    else:
        kw["start_new_session"] = True
    proc = subprocess.Popen(wrapper, **kw)
    meta = _read_meta(kind, jobs_dir) or {}
    meta["pid"] = proc.pid
    _write_meta(kind, meta, jobs_dir)
    return meta


def stop(kind: str, jobs_dir: Path = JOBS_DIR) -> bool:
    """Stop a running job and everything it started. Returns False if it was not running."""
    st = status(kind, jobs_dir)
    if st["state"] != "running":
        return False
    pid = int(st["meta"]["pid"])
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True)
    else:
        try:
            os.killpg(pid, signal.SIGTERM)
        except OSError:
            pass
    meta = _read_meta(kind, jobs_dir) or {}
    meta.update({"stopped": True, "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
    _write_meta(kind, meta, jobs_dir)
    _, log = _paths(kind, jobs_dir)
    with log.open("a", encoding="utf-8") as fh:
        fh.write("\n[stopped from the dashboard]\n")
    return True


def _run_wrapper(kind: str, jobs_dir: Path) -> int:
    """Run the job's command, append its output to the log, then record the exit code."""
    meta = _read_meta(kind, jobs_dir) or {}
    _, log = _paths(kind, jobs_dir)
    with log.open("a", encoding="utf-8") as fh:
        rc = subprocess.call(meta["cmd"], cwd=str(ROOT), stdout=fh, stderr=subprocess.STDOUT,
                             env=dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8"))
    meta = _read_meta(kind, jobs_dir) or meta
    if not meta.get("stopped"):
        meta.update({"returncode": rc, "ended": time.strftime("%Y-%m-%d %H:%M:%S")})
        _write_meta(kind, meta, jobs_dir)
    return rc


if __name__ == "__main__":
    sys.exit(_run_wrapper(sys.argv[1], Path(sys.argv[2])))
