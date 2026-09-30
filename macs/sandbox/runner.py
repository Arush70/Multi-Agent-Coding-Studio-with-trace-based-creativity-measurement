"""Isolated execution of generated code against a list of tests.

Arush - the model writes code and you must run it to know if it works, but that code is
UNTRUSTED (it could loop forever, eat all RAM, touch the network). So it never runs in
your normal process. Two backends:

  * docker  (THE EXPERIMENT): python:3.11-slim with --network none, memory/CPU caps, a
    read-only filesystem and a non-root user. A real security boundary. Every number in
    your results must come from this backend.
  * subprocess (DEVELOPMENT ONLY): the same harness in a child process, killed by a
    wall-clock timeout if it hangs. This is NOT a security boundary - it does not cap
    memory on Windows - so never run the real experiment on it, and say so in your
    methods. It exists so you can develop on Windows before Docker is working, and so CI
    (which has no guaranteed Docker) can test the rest of the pipeline.

Cross-platform note (why this changed): the first version set POSIX resource limits and
so only ran on Linux/mac. This version runs the harness as a child process with a
timeout on EVERY OS, and additionally applies CPU/memory limits only where the OS
supports them (POSIX). On Windows you still get process isolation and the timeout guard,
which is enough for development; Docker remains the backend for real runs.

The harness never GUESSES: if the code cannot run (crash/timeout), the result is an
explicit error, never a silent "0 passed" you might mistake for "the code is wrong".
"""
from __future__ import annotations

import hashlib, json, os, shutil, subprocess, sys, tempfile, time
from dataclasses import dataclass, asdict
from pathlib import Path

HARNESS = r'''
import json, sys, time, traceback
src = open(sys.argv[1]).read()
tests = json.load(open(sys.argv[2]))
results = []
ns = {}
try:
    exec(compile(src, "candidate.py", "exec"), ns)
    load_ok = True
except Exception:
    load_ok = False
    results = [{"i": i, "ok": False, "error": "candidate failed to load: " + traceback.format_exc()[-400:]} for i in range(len(tests))]
if load_ok:
    for i, t in enumerate(tests):
        local = dict(ns)              # fresh namespace per test
        t0 = time.perf_counter()
        try:
            exec(compile(t, f"test_{i}", "exec"), local)
            results.append({"i": i, "ok": True, "t": time.perf_counter() - t0})
        except AssertionError:
            results.append({"i": i, "ok": False, "error": "assertion failed", "t": time.perf_counter() - t0})
        except Exception:
            results.append({"i": i, "ok": False, "error": traceback.format_exc()[-400:], "t": time.perf_counter() - t0})
print("__MACS_RESULT__" + json.dumps(results))
'''


# Harness for HEURISTIC (Tier 3) scoring. Same idea as the test harness, but instead of
# pass/fail it runs a driver that exercises the candidate's function over frozen instances and
# returns a NUMBER (bins used, tour length, colours, makespan, cap-set size). The driver runs in
# the candidate's namespace with the instance data in `__data__` and must set `__metric__`.
# Arush - reusing the exact same sandbox (docker --network none etc.) means the hard/open Tier 3
# problems get scored under the same security boundary as everything else; only the harness differs.
METRIC_HARNESS = r'''
import json, sys, traceback
src = open(sys.argv[1]).read()
driver = open(sys.argv[2]).read()
data = json.load(open(sys.argv[3]))
ns = {}
out = {"ok": False}
try:
    exec(compile(src, "candidate.py", "exec"), ns)      # defines the candidate's entry point
    ns["__data__"] = data                               # frozen instances the driver runs on
    exec(compile(driver, "driver.py", "exec"), ns)      # driver must set ns["__metric__"]
    out = {"ok": True, "metric": ns.get("__metric__"), "valid": ns.get("__valid__", True),
           "detail": ns.get("__detail__")}
except Exception:
    out = {"ok": False, "error": traceback.format_exc()[-600:]}
print("__MACS_METRIC__" + json.dumps(out))
'''


@dataclass
class MetricResult:
    code_hash: str
    ok: bool                 # did candidate+driver run to completion (not a crash/timeout)?
    valid: bool              # did the candidate produce a FEASIBLE answer (driver's judgement)?
    metric: float | None     # the raw metric (direction depends on the task)
    detail: object           # optional per-instance breakdown from the driver
    runtime_s: float
    timed_out: bool
    error: str | None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RunResult:
    code_hash: str
    tests_passed: int
    tests_failed: int
    runtime_s: float
    timed_out: bool
    error: str | None
    per_test: list[dict]

    def as_trace_fields(self, test_set: str) -> dict:
        # Exactly the fields a `verification` trace event needs; keeps sandbox and schema aligned.
        return {
            "code_hash": self.code_hash,
            "tests_passed": self.tests_passed,
            "tests_failed": self.tests_failed,
            "test_set": test_set,
            "runtime_s": round(self.runtime_s, 4),
        }

    def to_dict(self) -> dict:
        return asdict(self)


def code_hash(src: str) -> str:
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _parse(stdout: str, n_tests: int) -> list[dict]:
    marker = "__MACS_RESULT__"
    for line in stdout.splitlines()[::-1]:
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return [{"i": i, "ok": False, "error": "no result marker (crash or timeout)"} for i in range(n_tests)]


def _posix_limits(cpu_seconds: int, memory_mb: int):
    # Returns a preexec function that caps CPU and address space - POSIX only.
    def _apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds))
        resource.setrlimit(resource.RLIMIT_AS, (memory_mb * 1024 * 1024, memory_mb * 1024 * 1024))
    return _apply


def run(
    src: str,
    tests: list[str],
    *,
    backend: str = "docker",
    cpu_seconds: int = 10,
    memory_mb: int = 512,
    image: str = "python:3.11-slim",
) -> RunResult:
    if backend == "docker" and shutil.which("docker") is None:
        raise RuntimeError("docker not found; install Docker or use backend='subprocess' (development only)")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "candidate.py").write_text(src)
        (d / "tests.json").write_text(json.dumps(tests))
        (d / "harness.py").write_text(HARNESS)

        preexec = None
        if backend == "docker":
            cmd = [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
                "--cpus", "1",
                "--pids-limit", "64",
                "--read-only", "--tmpfs", "/tmp:size=16m",
                "-v", f"{d}:/work:ro", "-w", "/work",
                "--user", "65534:65534",
                image, "python", "harness.py", "candidate.py", "tests.json",
            ]
            timeout = cpu_seconds + 5   # docker adds start-up overhead
        elif backend == "subprocess":
            # Run the harness file directly - works on every OS. The subprocess timeout
            # below is the cross-platform guard against an infinite loop. On POSIX we also
            # add CPU/memory rlimits for good measure.
            cmd = [sys.executable, "harness.py", "candidate.py", "tests.json"]
            timeout = cpu_seconds + 2
            if os.name == "posix":
                preexec = _posix_limits(cpu_seconds, memory_mb)
        else:
            raise ValueError(f"unknown backend {backend!r}")

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                cmd, cwd=d, capture_output=True, text=True, timeout=timeout,
                preexec_fn=preexec,   # None on Windows/docker; the rlimit fn on POSIX subprocess
            )
            stdout, stderr, timed_out = proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, timed_out = "timeout", True
        runtime = time.perf_counter() - t0

    per_test = _parse(stdout, len(tests))
    passed = sum(1 for r in per_test if r.get("ok"))
    return RunResult(
        code_hash=code_hash(src),
        tests_passed=passed,
        tests_failed=len(tests) - passed,
        runtime_s=runtime,
        # error is set only for a RIG failure (timeout / harness never produced a result),
        # never for a mere failed assertion, which is a normal wrong answer and is DATA.
        error=(stderr[-400:] or "killed (cpu/memory limit or crash)") if (timed_out or any("no result marker" in r.get("error", "") for r in per_test)) else None,
        timed_out=timed_out,
        per_test=per_test,
    )


def _parse_metric(stdout: str) -> dict:
    marker = "__MACS_METRIC__"
    for line in stdout.splitlines()[::-1]:
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    return {"ok": False, "error": "no metric marker (crash or timeout)"}


def run_metric(
    src: str,
    driver: str,
    data: object,
    *,
    backend: str = "docker",
    cpu_seconds: int = 15,
    memory_mb: int = 512,
    image: str = "python:3.11-slim",
) -> MetricResult:
    """Run a candidate heuristic over frozen `data` via `driver` and return the metric it computes.

    Arush - `driver` is a small trusted snippet (from macs/analysis/heuristic_score.py) that knows
    how to exercise this task's function and set `__metric__`/`__valid__`. `src` (the model's code)
    is still untrusted and runs under the same sandbox as the test runner.
    """
    if backend == "docker" and shutil.which("docker") is None:
        raise RuntimeError("docker not found; install Docker or use backend='subprocess' (development only)")

    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "candidate.py").write_text(src)
        (d / "driver.py").write_text(driver)
        (d / "data.json").write_text(json.dumps(data))
        (d / "harness.py").write_text(METRIC_HARNESS)

        preexec = None
        if backend == "docker":
            cmd = [
                "docker", "run", "--rm", "--network", "none",
                "--memory", f"{memory_mb}m", "--memory-swap", f"{memory_mb}m",
                "--cpus", "1", "--pids-limit", "64",
                "--read-only", "--tmpfs", "/tmp:size=16m",
                "-v", f"{d}:/work:ro", "-w", "/work",
                "--user", "65534:65534",
                image, "python", "harness.py", "candidate.py", "driver.py", "data.json",
            ]
            timeout = cpu_seconds + 5
        elif backend == "subprocess":
            cmd = [sys.executable, "harness.py", "candidate.py", "driver.py", "data.json"]
            timeout = cpu_seconds + 2
            if os.name == "posix":
                preexec = _posix_limits(cpu_seconds, memory_mb)
        else:
            raise ValueError(f"unknown backend {backend!r}")

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout, preexec_fn=preexec)
            stdout, stderr, timed_out = proc.stdout, proc.stderr, False
        except subprocess.TimeoutExpired as e:
            stdout = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
            stderr, timed_out = "timeout", True
        runtime = time.perf_counter() - t0

    parsed = _parse_metric(stdout)
    ok = bool(parsed.get("ok")) and not timed_out
    # A candidate that ran but produced an infeasible answer is valid=False with metric=None:
    # that is DATA (a bad heuristic), not a rig failure. error is set only for crash/timeout.
    return MetricResult(
        code_hash=code_hash(src),
        ok=ok,
        valid=bool(parsed.get("valid")) if ok else False,
        metric=(parsed.get("metric") if ok else None),
        detail=parsed.get("detail"),
        runtime_s=runtime,
        timed_out=timed_out,
        error=None if ok else (parsed.get("error") or stderr[-400:] or "killed (cpu/memory limit or crash)"),
    )
