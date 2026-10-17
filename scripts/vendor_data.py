#!/usr/bin/env python3
"""Vendor a data snapshot from tns-context-engine into src/tnscope/_data."""
import shutil
import subprocess
import sys
from pathlib import Path

DIRS = ["lexicon/hate", "lexicon/defamation", "normalize", "context"]
FILES = ["scoring/scoring.yml", "reference/meta.yml", "schemas/detection.schema.json"]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: vendor_data.py <data_repo_path>", file=sys.stderr)
        return 2
    src = Path(argv[1]).resolve()
    dest = Path(__file__).resolve().parent.parent / "src" / "tnscope" / "_data"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for d in DIRS:
        shutil.copytree(src / d, dest / d)
    for f in FILES:
        (dest / f).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / f, dest / f)
    sha = subprocess.check_output(["git", "-C", str(src), "rev-parse", "HEAD"], text=True).strip()
    (dest / "SOURCE").write_text(sha + "\n")
    print(f"vendored {sha} -> {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
