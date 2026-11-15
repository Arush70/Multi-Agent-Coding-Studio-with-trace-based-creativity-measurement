"""Exact compute time of the runs, from the traces and (on ISCA) from Slurm's accounting.

    python scripts/compute_time.py                                  # traces only
    python scripts/compute_time.py --sacct data/jobs/sacct_all.txt  # plus GPU hours per job

From the traces (data/traces and data/archive): per model, the number of sessions, the summed
session time, the time the model was busy (union of session intervals) and the median session.
From Slurm (`sacct ... -P`, see REPRODUCE.md): per job, the GPUs allocated (type and count) and the
elapsed time; each job is assigned to its model profile from its job log
(`data/jobs/isca-<jobid>.log`, line "== job <id> ... profile=<p>") or its job name ("m-<profile>").
GPU hours = elapsed hours x GPUs allocated, reported separately for full A100s and MIG slices.

Writes data/results/compute_time.json and docs/COMPUTE_TIME.md.
"""
from __future__ import annotations

import argparse, csv, datetime as dt, json, re, statistics as st
from collections import defaultdict
from pathlib import Path


COMPATIBLE_SINCE = (0, 4, 0)          # macs.orchestrator.COMPATIBLE_SINCE: older traces are not analysed


def _ver(v):
    try:
        return tuple(int(x) for x in str(v or "0.1.0").split(".")[:3])
    except ValueError:
        return (0, 0, 0)


def _ends(p: Path):
    try:
        with p.open("rb") as fh:                         # first and last line only (traces are large)
            a = json.loads(fh.readline())
            fh.seek(0, 2); size = fh.tell(); fh.seek(max(0, size - 4096))
            b = json.loads(fh.read().decode("utf-8", "ignore").strip().splitlines()[-1])
    except Exception:  # noqa: BLE001 - unreadable or empty trace
        return None
    if a.get("event") != "session_start" or b.get("event") != "session_end":
        return None
    h = a.get("header", {})
    return {"model": h.get("model", "?"), "archive": "archive" in p.parts, "name": p.name,
            "key": (h.get("model"), a.get("task_id"), a.get("condition"), h.get("seed")),
            "legacy": _ver(h.get("orchestrator_version")) < COMPATIBLE_SINCE,
            "complete": str(b.get("content", "")).startswith(("completed", "stopped")),
            "t0": dt.datetime.fromisoformat(a["ts"]), "t1": dt.datetime.fromisoformat(b["ts"])}


def from_traces(dirs):
    """Same selection as macs.analysis.aggregate: in data/traces, current orchestrator version,
    completed, and the earliest trace per (model, task, condition, seed); everything else is listed
    as 'excluded' (older versions, duplicates, crashed) or 'archive' (data/archive)."""
    from concurrent.futures import ThreadPoolExecutor
    files = [p for d in dirs for p in Path(d).rglob("*.jsonl")]
    with ThreadPoolExecutor(32) as ex:                   # file opens dominate on network/synced drives
        recs = [r for r in ex.map(_ends, files) if r]
    first = {}
    for r in sorted((r for r in recs if not r["archive"] and r["complete"]), key=lambda r: r["t0"]):
        first.setdefault(r["key"], r["name"])
    per = defaultdict(list)
    for r in recs:
        if r["archive"]:
            where = "archive"
        elif r["legacy"] or not r["complete"] or first.get(r["key"]) != r["name"]:
            where = "excluded"
        else:
            where = "analysed"
        per[(r["model"], where)].append((r["t0"], r["t1"]))
    out = {}
    for (m, where), iv in per.items():
        iv.sort()
        busy, (cs, ce) = 0.0, iv[0]
        for s, e in iv[1:]:
            if s > ce:
                busy += (ce - cs).total_seconds(); cs, ce = s, e
            else:
                ce = max(ce, e)
        busy += (ce - cs).total_seconds()
        mins = [(e - s).total_seconds() / 60 for s, e in iv]
        out.setdefault(m, {})[where] = {
            "sessions": len(iv), "session_hours": round(sum(mins) / 60, 1), "busy_hours": round(busy / 3600, 1),
            "median_session_min": round(st.median(mins), 1),
            "first": iv[0][0].isoformat(timespec="minutes"), "last": max(e for _, e in iv).isoformat(timespec="minutes")}
    return out


def _hours(elapsed: str) -> float:
    days, _, rest = elapsed.rpartition("-")
    h, m, s = (int(x) for x in rest.split(":"))
    return (int(days) if days else 0) * 24 + h + m / 60 + s / 3600


def from_sacct(path, logs_dir):
    prof_of = {}
    for p in Path(logs_dir).glob("*.log"):
        m = re.search(r"== job (\d+) on \S+ .*?profile=(\S+)", p.read_text(encoding="utf-8", errors="ignore"))
        if m:
            prof_of[m.group(1)] = m.group(2)
    rows = list(csv.DictReader(Path(path).read_text(encoding="utf-8").splitlines(), delimiter="|"))
    agg = defaultdict(lambda: defaultdict(float))
    jobs = []
    for r in rows:
        if "." in r["JobID"]:                     # job steps (.batch, .extern) repeat the job
            continue
        jid = r["JobID"]
        if not r.get("Elapsed") or r["Elapsed"] in ("00:00:00", ""):
            continue
        tres = r.get("AllocTRES", "")
        g = re.search(r"gres/gpu(?::([\w.]+))?=(\d+)", tres)
        if not g:
            continue
        kind = g.group(1) or r.get("Partition", "gpu")
        kind = "MIG 1g.10gb" if "1g" in kind or "mig" in r.get("Partition", "") else \
               "V100" if "volta" in r.get("Partition", "") or "v100" in kind.lower() else "A100 80GB"
        n = int(g.group(2))
        name = r.get("JobName", "")
        prof = prof_of.get(jid) or (re.match(r"m-([a-z0-9_.]+?)(?:-g\d.*)?$", name) or [None, name])[1]
        hrs = _hours(r["Elapsed"])
        agg[prof][f"{kind} GPU-hours"] += hrs * n
        agg[prof]["job wall hours"] += hrs
        jobs.append({"job": jid, "name": name, "profile": prof, "partition": r.get("Partition"), "gpus": n,
                     "gpu": kind, "state": r.get("State"), "elapsed_h": round(hrs, 2)})
    return {p: {k: round(v, 1) for k, v in d.items()} for p, d in agg.items()}, jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", nargs="+", default=["data/traces", "data/archive"])
    ap.add_argument("--sacct", default=None)
    ap.add_argument("--logs", default="data/jobs")
    a = ap.parse_args()
    tr = from_traces(a.traces)
    res = {"traces": tr}
    md = ["# Compute time", "", "Generated by `scripts/compute_time.py` from the traces"
          + (" and Slurm accounting (`sacct`)." if a.sacct else "."), "",
          "## Sessions per model (analysed traces)", "",
          "Analysed = the sessions in the reported analysis (current version, completed, one per cell, "
          "before the cut-off). Model busy = time at least one session of the model was running.", "",
          "| Model | Sessions | Median session (min) | Summed session time (h) | Model busy (h) | First | Last |",
          "|---|---|---|---|---|---|---|"]
    for m, d in sorted(tr.items()):
        x = d.get("analysed")
        if x:
            md.append(f"| {m} | {x['sessions']:,} | {x['median_session_min']} | {x['session_hours']} | {x['busy_hours']} | {x['first']} | {x['last']} |")
    tot = sum(d["analysed"]["sessions"] for d in tr.values() if "analysed" in d)
    md.append(f"| **total** | {tot:,} | | {sum(d['analysed']['session_hours'] for d in tr.values() if 'analysed' in d):.1f} | | | |")
    if a.sacct:
        agg, jobs = from_sacct(a.sacct, a.logs)
        res["slurm"], res["jobs"] = agg, jobs
        kinds = sorted({k for d in agg.values() for k in d if k.endswith("GPU-hours")})
        md += ["", "## GPU hours per profile (Slurm accounting)", "",
               "GPUs allocated x elapsed time, for every job (including failed, cancelled and restarted "
               "jobs and time spent loading models), so this is the GPU time used, not only time in sessions.", "",
               "| Profile | " + " | ".join(kinds) + " | Job wall hours |", "|---" * (len(kinds) + 2) + "|"]
        tot = defaultdict(float)
        for p, d in sorted(agg.items()):
            md.append(f"| {p} | " + " | ".join(str(d.get(k, 0)) for k in kinds) + f" | {d.get('job wall hours', 0)} |")
            for k in kinds + ["job wall hours"]:
                tot[k] += d.get(k, 0)
        md.append("| **total** | " + " | ".join(f"{tot[k]:.1f}" for k in kinds) + f" | {tot['job wall hours']:.1f} |")
    Path("data/results").mkdir(parents=True, exist_ok=True)
    Path("data/results/compute_time.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    Path("docs/COMPUTE_TIME.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    main()
