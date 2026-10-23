"""Write config/models.yaml from isca/models.isca.yaml with each model's revision pinned from
`ollama list` (the running server is found through OLLAMA_HOST). The file is replaced in one step,
so jobs that are already running never read a half-written config.

    python isca/pin_revisions.py                 # uses the ollama binary on PATH or $OLLAMA
    python isca/pin_revisions.py --model gemma3:12b   # print that model's profile name and exit
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "isca" / "models.isca.yaml"
CONFIG = ROOT / "config" / "models.yaml"


def ollama_ids(ollama: str) -> dict[str, str]:
    out = subprocess.run([ollama, "list"], capture_output=True, text=True, check=True).stdout
    ids = {}
    for line in out.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2:
            ids[parts[0]] = parts[1][:12]
    return ids


def pin(text: str, ids: dict[str, str]) -> tuple[str, list[str]]:
    out, model, done = [], None, []
    for line in text.splitlines():
        m = re.match(r"\s+model:\s*(\S+)", line)
        if m:
            model = m.group(1)
        r = re.match(r'(\s+revision:\s*)""', line)
        if r and model in ids:
            line = f'{r.group(1)}"{ids[model]}"'
            done.append(f"{model} -> {ids[model]}")
        out.append(line)
    return "\n".join(out) + "\n", done


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", default=os.environ.get("OLLAMA", "ollama"))
    ap.add_argument("--profile-model", metavar="PROFILE", help="print the model name of PROFILE and exit")
    args = ap.parse_args()
    if args.profile_model:
        import yaml
        print(yaml.safe_load(TEMPLATE.read_text())["profiles"][args.profile_model]["model"])
        return
    text, done = pin(TEMPLATE.read_text(encoding="utf-8"), ollama_ids(args.ollama))
    CONFIG.parent.mkdir(exist_ok=True)
    tmp = CONFIG.with_suffix(".yaml.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, CONFIG)
    for d in done:
        print("pinned", d)


if __name__ == "__main__":
    sys.exit(main())
