"""Isolated execution of generated code against a list of tests.

Arush - the model writes code, and you have to run it to know if it works. But that
code is UNTRUSTED: it might loop forever, allocate all your RAM, delete files, or try
to reach the network. So it never runs in your normal process. Two backends:

  * docker  (THE EXPERIMENT): runs python:3.11-slim with --network none, a memory cap,
    a CPU cap, a read-only filesystem and a non-root user. This is a real security
    boundary and it is what every number in your results must come from.
  * subprocess (DEVELOPMENT ONLY): the same test harness in a child process with
    resource limits. Convenient on your laptop while Docker is being set up, but it is
    NOT a security boundary - do not run the real experiment on it, and say so in your
    methods. It exists so you (and CI, which has no guaranteed Docker) can test the
    rest of the pipeline.

Tests are plain assert expressions over the entry-point function, e.g.
"assert two_sum([2,7,11,15],9)==[0,1]". Each test runs in its OWN namespace so one
failing test cannot corrupt the next.

Design rule: the harness never GUESSES. If the sandbox cannot run the code (crash,
timeout, killed by a limit), the result is an explicit error, never a silent "0 passed"
that you might mistake for "the code is wrong". Distinguishing "wrong answer" from
"never ran" matters: the first is data about the model, the second is a bug in your rig.
"""
from __future__ import annotations

import hashlib, json, shutil, subprocess, sys, tempfile, textwrap, time
from dataclasses import dataclass, asdict
from pathlib import Path

# This runs INSIDE the sandbox. It loads the candidate, runs each test in a fresh copy
# of the candidate's namespace, and prints a single machine-readable result line.
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
        # Exactly the fields a `verification` trace event needs. Keeping this here means
        # the sandbox and the trace schema can never drift apart.
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
    # Short content hash of the candidate; logged so identical code is recognisable
    # across sessions and you can dedupe solutions later.
    return hashlib.sha256(src.encode()).hexdigest()[:16]


def _parse(stdout: str, n_tests: int) -> list[dict]:
    marker = "__MACS_RESULT__"
    for line in stdout.splitlines()[::-1]:
        if line.startswith(marker):
            return json.loads(line[len(marker):])
    # No marker => the process died before finishing (crash/timeout/OOM). Every test is
    # an explicit "never ran", not a silent fail.
    return [{"i": i, "ok": False, "error": "no result marker (crash or timeout)"} for i in range(n_tests)]


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

        if backend == "docker":
            # Each flag closes a specific hole:
            #   --network none      : the code cannot phone home or fetch anything
            #   --memory / swap     : an allocation bomb is killed, not your laptop
            #   --cpus / pids-limit : a fork bomb or busy loop is contained
            #   --read-only + tmpfs : the code cannot write anywhere except a small /tmp
            #   --user 65534        : runs as 'nobody', not root
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
            timeout = cpu_seconds + 5   # wall-clock allowance; docker adds start-up overhead
        elif backend == "subprocess":
            # Dev-only: resource limits via the 'resource' module (POSIX). On your Windows
            # laptop the limits are best-effort; the real runs use docker anyway.
            preexec = textwrap.dedent(f"""
                import resource, sys
                resource.setrlimit(resource.RLIMIT_CPU, ({cpu_seconds}, {cpu_seconds}))
                resource.setrlimit(resource.RLIMIT_AS, ({memory_mb}*1024*1024, {memory_mb}*1024*1024))
                sys.argv = ["harness.py", "candidate.py", "tests.json"]
                exec(open("harness.py").read())
            """)
            cmd = [sys.executable, "-c", preexec]
            timeout = cpu_seconds + 2
        else:
            raise ValueError(backend)

        t0 = time.perf_counter()
        try:
            proc = subprocess.run(cmd, cwd=d, capture_output=True, text=True, timeout=timeout)
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
        timed_out=timed_out,
        # An error is set when the run itself failed (timeout, or the harness never
        # produced a result), NOT merely when a test's assertion failed - that is a
        # normal "wrong answer", which is data, not an error.
        error=(stderr[-400:] or "killed (cpu/memory limit or crash)") if (timed_out or any("no result marker" in r.get("error", "") for r in per_test)) else None,
        per_test=per_test,
    )
