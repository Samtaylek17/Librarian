#!/usr/bin/env python3
"""Verify vendored files in third_party/ are byte-identical to upstream
at the pinned commit. Exits non-zero on drift.

Usage:
    python scripts/check_third_party_drift.py
"""
from __future__ import annotations

import argparse
import base64
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# (vendored_relpath, github_repo, upstream_relpath)
PINS: list[tuple[str, str, str]] = [
    (
        "third_party/openhands_vendored/stuck.py",
        "All-Hands-AI/OpenHands",
        "openhands/controller/stuck.py",
    ),
    (
        "third_party/openhands_vendored/prompts/system_prompt.j2",
        "All-Hands-AI/OpenHands",
        "openhands/agenthub/codeact_agent/prompts/system_prompt.j2",
    ),
]


def read_pin(repo_dir: str) -> str:
    p = REPO / repo_dir / "COMMIT_PIN.txt"
    if not p.exists():
        sys.exit(f"missing COMMIT_PIN.txt in {repo_dir}")
    return p.read_text().strip()


def upstream_content(repo: str, path: str, commit: str) -> str:
    out = subprocess.run(
        ["gh", "api", f"/repos/{repo}/contents/{path}?ref={commit}", "--jq", ".content"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    return base64.b64decode(out).decode("utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    drift_count = 0
    for vendored_rel, repo, upstream_rel in PINS:
        vendored_path = REPO / vendored_rel
        if not vendored_path.exists():
            print(f"  MISSING: {vendored_rel}")
            drift_count += 1
            continue
        repo_dir = vendored_rel.split("/", 2)[1]  # e.g., openhands_vendored
        commit = read_pin(f"third_party/{repo_dir}")
        try:
            up = upstream_content(repo, upstream_rel, commit)
        except subprocess.CalledProcessError as e:
            print(f"  FETCH FAIL: {vendored_rel} from {repo}@{commit[:8]}")
            drift_count += 1
            continue
        local = vendored_path.read_text()
        if up != local:
            print(f"  DRIFT: {vendored_rel} differs from {repo}@{commit[:8]}:{upstream_rel}")
            drift_count += 1
        elif not args.quiet:
            print(f"  OK: {vendored_rel} ({repo}@{commit[:8]})")

    if drift_count:
        print(f"\n{drift_count} file(s) drifted from their pinned upstream content.")
        return 1
    print("\nAll vendored files match their pinned commit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
