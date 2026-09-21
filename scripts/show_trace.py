"""Write one example session and print it back, validated.

Arush - run this to SEE a trace with your own eyes before the orchestrator exists:

    python scripts/show_trace.py

It writes a small condition-C session to data/traces/example.jsonl, validates it,
and prints each line. This is the shape every real session will take - learner
turns, a tutor question tagged QUESTION (ASTRA-style), a proposal, a verification.
"""
from pathlib import Path
from macs.trace.writer import TraceWriter, validate_file, read_trace

HEADER = {
    "model": "qwen2.5-coder:7b-instruct-q4_K_M",
    "revision": "dae161e27b0e",
    "quantisation": "q4_K_M",
    "temperature": 0.7,
    "seed": 1,
    "prompt_hashes": {"tutor": "abc123"},
}


def main() -> None:
    path = Path("data/traces/example.jsonl")
    if path.exists():
        path.unlink()   # start clean each run
    with TraceWriter(path, session_id="S_demo", task_id="t1_two_sum", condition="C", header=HEADER) as tr:
        tr.log("learner_1", "turn", content="I'd check every pair of numbers.")
        tr.log("tutor", "turn", content="What is the time cost of checking every pair?", action_tag="QUESTION")
        tr.log("learner_2", "turn", content="Maybe store what we've seen in a dict?")
        tr.log("tutor", "turn", content="Good - try that.", action_tag="ENCOURAGEMENT")
        tr.log("tutor", "proposal", content="def two_sum(nums, t): seen={}; ...", proposal_id="p1")
        tr.log("verifier", "verification", content="public tests", code_hash="deadbeef",
               tests_passed=2, tests_failed=0, test_set="public", runtime_s=0.008)

    errors = validate_file(path)
    print(f"validation: {'OK, 0 errors' if not errors else errors}")
    print(f"--- {path} ---")
    for ev in read_trace(path):
        print(f"seq {ev['seq']:>2}  {ev['role']:<10} {ev['event']:<14} "
              f"{ev.get('action_tag',''):<12} {ev.get('content','')[:50]}")


if __name__ == "__main__":
    main()
