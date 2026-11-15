"""Build the complete submission package: code at a git tag plus every data file needed to check
or repeat the results.

    python scripts/package_submission.py --tag dissertation-submission-v1.0 --out ../Submission_code
    python scripts/package_submission.py --tag dissertation-submission-v1.0 --out ../Submission_code \\
        --data-tgz macs_data_final.tgz        # read the bulk data from the archive copied from ISCA

Contents: the tracked code at the tag (git archive), then data/traces, data/archive (traces after
the cut-off and earlier pilots), data/reference (every model's reference sets), data/cache (the
response cache: replays every LLM call exactly), data/results, data/jobs (job logs and the Slurm
accounting file), data/calibration, and the rating study without personal data (pairs, items,
booklet, AI rater sheets and answers, aggregate results, responses under rater codes only).
Never included: config/models.yaml (may hold keys) and the Forms exports with timestamps.

The build can be stopped and run again: it reopens the zip and skips files already in it
(--budget limits one run to that many seconds). When everything is in, it writes
<name>.zip.sha256 and <name>_MANIFEST.txt (every file with its size and SHA-256).
"""
from __future__ import annotations

import argparse, hashlib, io, subprocess, sys, tarfile, time, zipfile
from pathlib import Path

DATA = ["data/traces", "data/archive", "data/reference", "data/cache", "data/results", "data/jobs"]
LOCAL = ["data/calibration"] + [f"data/rating/{r}" for r in (
    "pairs.json", "label_items.json", "rating_booklet.html", "results.json", "judge.json",
    "responses.json", "ai_raters", "human_study_v1.0")]
NEVER = ("config/models.yaml", "form_responses", ".lock", "__pycache__", "_to_delete", "/.git/")


class Budget(Exception):
    pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--data-tgz", default=None, help="tar.gz of data/ made on ISCA (faster than many small files)")
    ap.add_argument("--no-cache", action="store_true", help="leave out data/cache")
    ap.add_argument("--budget", type=float, default=0, help="stop after this many seconds (0 = no limit)")
    a = ap.parse_args()
    t0 = time.time()
    name = f"MACS_{a.tag}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{name}.zip"
    data = [d for d in DATA if not (a.no_cache and d == "data/cache")]
    have = set()
    if zpath.exists():
        with zipfile.ZipFile(zpath) as z:
            have = set(z.namelist())
    added = 0

    def put(z, rel, b):
        nonlocal added
        arc = f"{name}/{rel}"
        if arc in have or any(x in "/" + rel for x in NEVER):
            return
        z.writestr(arc, b); have.add(arc); added += 1
        if a.budget and time.time() - t0 > a.budget:
            raise Budget

    try:
        with zipfile.ZipFile(zpath, "a", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
            code = subprocess.run(["git", "archive", "--format=tar", a.tag], capture_output=True, check=True).stdout
            with tarfile.open(fileobj=io.BytesIO(code)) as t:
                for m in t.getmembers():
                    if m.isfile():
                        put(z, m.name, t.extractfile(m).read())
            if a.data_tgz:
                with tarfile.open(a.data_tgz, "r|gz") as t:          # streamed: one pass, no seeking
                    for m in t:
                        rel = m.name.lstrip("./")
                        if m.isfile() and any(rel.startswith(d + "/") for d in data):
                            if f"{name}/{rel}" in have:
                                continue
                            put(z, rel, t.extractfile(m).read())
            else:
                for d in data:
                    for p in sorted(x for x in Path(d).rglob("*") if x.is_file()):
                        if f"{name}/{p.as_posix()}" not in have:
                            put(z, p.as_posix(), p.read_bytes())
            for d in LOCAL:
                root = Path(d)
                for p in ([root] if root.is_file() else sorted(x for x in root.rglob("*") if x.is_file())):
                    put(z, p.as_posix(), p.read_bytes())
    except Budget:
        print(f"partial: {added} files added this run, {len(have)} in the zip; run the same command again")
        sys.exit(2)
    lines = []
    with zipfile.ZipFile(zpath) as z:
        for info in z.infolist():
            lines.append(f"{hashlib.sha256(z.read(info)).hexdigest()}  {info.file_size:>12}  {info.filename}\n")
    (out / f"{name}_MANIFEST.txt").write_text("".join(lines), encoding="utf-8")
    h = hashlib.sha256(zpath.read_bytes()).hexdigest()
    (out / f"{name}.zip.sha256").write_text(f"{h}  {zpath.name}\n", encoding="utf-8")
    print(f"complete: {zpath}: {len(lines)} files, {zpath.stat().st_size / 1e6:.1f} MB, sha256 {h}")


if __name__ == "__main__":
    main()
