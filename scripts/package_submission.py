"""Build the complete submission package: code at a git tag plus every data file needed to check
or repeat the results.

    python scripts/package_submission.py --tag dissertation-submission-v1.0 --out ../Submission_code

Contents: the tracked code at the tag (git archive), then data/traces, data/archive (traces after
the cut-off and earlier pilots), data/reference (every model's reference sets), data/cache (the
response cache: replays every LLM call exactly), data/results, data/calibration, data/jobs (job
logs and the Slurm accounting file), and the rating study without personal data (pairs, items,
booklet, AI rater sheets and answers, aggregate results, responses under rater codes only).
Never included: config/models.yaml (may hold keys) and the Forms exports with timestamps.
Writes <name>.zip, <name>.zip.sha256 and <name>_MANIFEST.txt (every file with its size and SHA-256).
"""
from __future__ import annotations

import argparse, hashlib, io, subprocess, tarfile, zipfile
from pathlib import Path

DATA = ["data/traces", "data/archive", "data/reference", "data/cache", "data/results",
        "data/calibration", "data/jobs"]
RATING = ["pairs.json", "label_items.json", "rating_booklet.html", "results.json", "judge.json",
          "responses.json", "ai_raters", "human_study_v1.0"]
NEVER = ("config/models.yaml", "form_responses", ".lock", "__pycache__", "_to_delete")


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", default="dist")
    ap.add_argument("--no-cache", action="store_true", help="leave out data/cache (large)")
    a = ap.parse_args()
    name = f"MACS_{a.tag}"
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    zpath = out / f"{name}.zip"
    manifest = []
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        code = subprocess.run(["git", "archive", "--format=tar", a.tag], capture_output=True, check=True).stdout
        with tarfile.open(fileobj=io.BytesIO(code)) as t:
            for m in t.getmembers():
                if m.isfile() and not any(x in m.name for x in NEVER):
                    b = t.extractfile(m).read()
                    z.writestr(f"{name}/{m.name}", b); manifest.append((m.name, len(b), sha(b)))
        done = {f for f, _, _ in manifest}
        paths = [Path(d) for d in DATA if not (a.no_cache and d == "data/cache")]
        paths += [Path("data/rating") / r for r in RATING]
        for root in paths:
            files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file())
            for p in files:
                rel = p.as_posix()
                if any(x in rel for x in NEVER) or rel in done:   # tracked files come from the tag
                    continue
                b = p.read_bytes()
                z.writestr(f"{name}/{rel}", b); manifest.append((rel, len(b), sha(b)))
    (out / f"{name}_MANIFEST.txt").write_text(
        "".join(f"{h}  {n:>12}  {f}\n" for f, n, h in manifest), encoding="utf-8")
    h = sha(zpath.read_bytes())
    (out / f"{name}.zip.sha256").write_text(f"{h}  {zpath.name}\n", encoding="utf-8")
    print(f"{zpath}: {len(manifest)} files, {zpath.stat().st_size / 1e6:.1f} MB, sha256 {h}")


if __name__ == "__main__":
    main()
