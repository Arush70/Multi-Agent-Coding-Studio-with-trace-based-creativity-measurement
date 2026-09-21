"""Inspect one task: show what an agent sees, and (separately) what the scorer sees.

Arush - run this on any task you write, to eyeball the boundary with your own eyes:

    python scripts/check_task.py tasks/tier1/t1_two_sum

The top block is agent_view - the ONLY thing the model is shown. The bottom block is
the held-out tests, which the model never sees. If anything from the bottom appears in
the top, stop: you have a leak.
"""
import sys
from macs.tasks.bank import load_task


def main() -> None:
    task_dir = sys.argv[1] if len(sys.argv) > 1 else "tasks/tier1/t1_two_sum"
    t = load_task(task_dir)
    print(f"=== TASK {t.id} (tier {t.tier}, scoring={t.scoring}) ===")
    print(f"known_families (blind labels): {t.known_families}")
    print("\n----- WHAT THE AGENT SEES (agent_view) -----")
    print(t.agent_view)
    print("\n----- WHAT ONLY THE SCORER SEES (held-out) -----")
    hts = t.heldout_tests()
    if not hts:
        print("(none - heuristic-scored task)")
    for h in hts:
        print(" ", h)

    # A tiny built-in leak check, so the script itself flags the disaster case.
    leaks = [h for h in hts if h in t.agent_view]
    print("\nLEAK CHECK:", "OK - no held-out test in agent_view" if not leaks else f"LEAK! {leaks}")


if __name__ == "__main__":
    main()
