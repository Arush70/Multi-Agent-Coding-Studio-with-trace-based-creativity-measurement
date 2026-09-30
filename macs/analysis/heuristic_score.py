"""Score Tier 3 HARD/OPEN problems by a metric instead of pass/fail.

Arush - Tier 1/2 tasks are 'did it pass the held-out tests?'. Tier 3 problems (online bin packing,
TSP construction, graph colouring, scheduling, cap sets) are NP-hard or open: there is no known
optimum to test against, so a solution is judged by HOW GOOD it is - fewer bins, shorter tour,
fewer colours, lower makespan, bigger cap set - measured on frozen instances and compared to a
standard baseline heuristic. This is exactly the setting FunSearch used, and it's where a creative
studio has the most room to beat the obvious approach.

For each task:
  * a DRIVER (trusted, below) runs the model's function over the frozen instances and returns the
    raw metric, plus whether the answer was feasible (an infeasible heuristic scores 0, it is not
    a crash);
  * a BASELINE (the named standard heuristic) is run through the SAME driver, so every score is
    'relative to best_fit / nearest_neighbour / ...';
  * the candidate's code (untrusted) runs in the sandbox, same boundary as the test runner.

`score` is normalised so HIGHER IS ALWAYS BETTER (score > 1 => beats the baseline), whatever the
raw metric's direction, so it drops straight into the per-condition table.
"""
from __future__ import annotations

import json
from functools import lru_cache

from macs.tasks.bank import Task
from macs.sandbox import runner

# ---------------------------------------------------------------- drivers (trusted)
# Each driver runs in the candidate's namespace with the instances in `__data__`. It must set
# __metric__ (float or None), __valid__ (bool) and may set __detail__. `globals()[ENTRY]` is the
# model's function; per-instance failures are caught so one bad instance can't crash the score.

_BIN_PACKING_DRIVER = r'''
fn = globals()[__data__["entry_point"]]
EPS = 1e-9
res = []
for items in __data__["instances"]:
    bins = []; ok = True
    for item in items:
        placed = False
        if bins:
            try:
                scores = fn(item, list(bins))
            except Exception:
                ok = False; break
            if not (isinstance(scores, (list, tuple)) and len(scores) == len(bins)):
                ok = False; break
            feasible = [i for i in range(len(bins)) if bins[i] >= item - EPS]
            if feasible:
                j = max(feasible, key=lambda i: scores[i]); bins[j] -= item; placed = True
        if not placed:
            bins.append(1.0 - item)
    res.append(len(bins) if ok else None)
vals = [r for r in res if r is not None]
__valid__ = len(vals) == len(res) and bool(vals)
__metric__ = (sum(vals) / len(vals)) if vals else None
__detail__ = res
'''

_SCHEDULING_DRIVER = r'''
fn = globals()[__data__["entry_point"]]
m = __data__["params"]["machines"]
res = []
for jobs in __data__["instances"]:
    loads = [0.0] * m; ok = True
    for job in jobs:
        try:
            scores = fn(job, list(loads))
        except Exception:
            ok = False; break
        if not (isinstance(scores, (list, tuple)) and len(scores) == m):
            ok = False; break
        j = max(range(m), key=lambda i: scores[i]); loads[j] += job
    res.append(max(loads) if ok else None)
vals = [r for r in res if r is not None]
__valid__ = len(vals) == len(res) and bool(vals)
__metric__ = (sum(vals) / len(vals)) if vals else None
__detail__ = res
'''

_TSP_DRIVER = r'''
fn = globals()[__data__["entry_point"]]
res = []
for dist in __data__["instances"]:
    n = len(dist); cur = 0; unvis = set(range(n)); unvis.discard(0); length = 0.0; ok = True
    while unvis:
        try:
            nxt = fn(cur, set(unvis), dist)
        except Exception:
            ok = False; break
        if nxt not in unvis:
            ok = False; break
        length += dist[cur][nxt]; unvis.discard(nxt); cur = nxt
    if ok:
        length += dist[cur][0]
    res.append(length if ok else None)
vals = [r for r in res if r is not None]
__valid__ = len(vals) == len(res) and bool(vals)
__metric__ = (sum(vals) / len(vals)) if vals else None
__detail__ = res
'''

_GRAPH_COLORING_DRIVER = r'''
fn = globals()[__data__["entry_point"]]
res = []
for g_raw in __data__["instances"]:
    graph = {int(k): set(int(x) for x in v) for k, v in g_raw.items()}
    try:
        order = fn({k: set(v) for k, v in graph.items()})
    except Exception:
        res.append(None); continue
    if not (isinstance(order, (list, tuple)) and sorted(order) == sorted(graph.keys())):
        res.append(None); continue
    color = {}
    for v in order:
        used = {color[u] for u in graph[v] if u in color}
        c = 0
        while c in used:
            c += 1
        color[v] = c
    res.append((max(color.values()) + 1) if color else 0)
vals = [r for r in res if r is not None]
__valid__ = len(vals) == len(res) and bool(vals)
__metric__ = (sum(vals) / len(vals)) if vals else None
__detail__ = res
'''

_CAP_SET_DRIVER = r'''
fn = globals()[__data__["entry_point"]]
def _is_cap_set(vs, n):
    S = set()
    for v in vs:
        if not (isinstance(v, (list, tuple)) and len(v) == n and all(x in (0, 1, 2) for x in v)):
            return False, S
        S.add(tuple(v))
    if len(S) != len(vs):
        return False, S          # duplicates are not allowed
    L = list(S)
    for i in range(len(L)):
        a = L[i]
        for j in range(i + 1, len(L)):
            b = L[j]
            c = tuple((-(a[t] + b[t])) % 3 for t in range(n))
            if c != a and c != b and c in S:
                return False, S    # a, b, c are three in a line
    return True, S
res = []
for n in __data__["params"]["dimensions"]:
    try:
        vs = fn(n)
    except Exception:
        res.append(None); continue
    ok, S = _is_cap_set(vs, n)
    res.append(len(S) if ok else 0)
vals = [r for r in res if r is not None]
__valid__ = len(vals) == len(res) and all(r > 0 for r in vals)
__metric__ = (sum(vals) / len(vals)) if vals else None
__detail__ = res
'''

# ---------------------------------------------------------------- baselines (same signature)
# The standard heuristic for each task, run through the SAME driver to give a reference metric.

_BASELINES = {
    "t3_bin_packing": "def priority(item, bins):\n    return [-(b - item) for b in bins]\n",          # best_fit
    "t3_scheduling": "def priority(job, loads):\n    return [-l for l in loads]\n",                    # least_loaded
    "t3_tsp_construction": "def choose_next(current, unvisited, dist):\n    return min(unvisited, key=lambda c: dist[current][c])\n",  # nearest_neighbour
    "t3_graph_coloring": "def vertex_order(graph):\n    return sorted(graph, key=lambda v: len(graph[v]), reverse=True)\n",           # largest_first
    "t3_cap_set": (
        "def build_cap_set(n):\n"
        "    import itertools\n"
        "    S = []; Sset = set()\n"
        "    for v in itertools.product((0, 1, 2), repeat=n):\n"
        "        ok = True\n"
        "        for a in S:\n"
        "            c = tuple((-(a[t] + v[t])) % 3 for t in range(n))\n"
        "            if c != a and c != v and c in Sset:\n"
        "                ok = False; break\n"
        "        if ok:\n"
        "            S.append(v); Sset.add(v)\n"
        "    return S\n"
    ),  # greedy_lexicographic
}

# task.id -> (driver, direction). direction = +1 if higher raw metric is better, -1 if lower is better.
_SPEC = {
    "t3_bin_packing": (_BIN_PACKING_DRIVER, -1),
    "t3_scheduling": (_SCHEDULING_DRIVER, -1),
    "t3_tsp_construction": (_TSP_DRIVER, -1),
    "t3_graph_coloring": (_GRAPH_COLORING_DRIVER, -1),
    "t3_cap_set": (_CAP_SET_DRIVER, +1),
}


def _build_data(task: Task) -> dict:
    """Assemble what the driver needs: the frozen instances + the task's parameters."""
    data: dict = {"entry_point": task.entry_point, "params": dict(task.heuristic)}
    inst_file = task.heuristic.get("instance_file")
    if inst_file:
        assert task.path is not None
        p = task.path / inst_file
        if not p.exists():
            raise FileNotFoundError(f"no frozen instances for {task.id}: run scripts/make_instances.py ({p})")
        data["instances"] = json.loads(p.read_text())["instances"]
    else:
        data["instances"] = []          # cap_set is instance-free (dimensions come from params)
    return data


@lru_cache(maxsize=None)
def _baseline_metric(task_id: str, backend: str, data_json: str) -> float | None:
    """Metric of the standard baseline heuristic on the same frozen instances (cached)."""
    driver = _SPEC[task_id][0]
    data = json.loads(data_json)
    r = runner.run_metric(_BASELINES[task_id], driver, data, backend=backend)
    return r.metric if (r.ok and r.valid) else None


def score_heuristic_code(task: Task, code: str, *, backend: str = "subprocess") -> dict:
    """Run one heuristic on the frozen instances and score it against the baseline.

    Returns raw `metric`, the `baseline` metric, a `valid` flag, and a normalised `score`
    (higher is always better; > 1 beats the baseline; 0 if infeasible; None if it crashed).
    """
    if task.id not in _SPEC:
        raise KeyError(f"no heuristic scorer registered for {task.id}")
    driver, direction = _SPEC[task.id]
    data = _build_data(task)
    r = runner.run_metric(code, driver, data, backend=backend)

    baseline = _baseline_metric(task.id, backend, json.dumps(data))
    score = None
    if r.ok and r.valid and r.metric is not None and baseline:
        # normalise so higher = better regardless of the raw metric's direction
        score = (r.metric / baseline) if direction > 0 else (baseline / r.metric)
    elif r.ok and not r.valid:
        score = 0.0                       # ran but infeasible -> worst score, still DATA
    return {
        "scoring": "heuristic",
        "ok": r.ok, "valid": bool(r.valid),
        "metric": r.metric, "baseline": baseline, "direction": direction,
        "score": score, "detail": r.detail, "error": r.error,
    }
