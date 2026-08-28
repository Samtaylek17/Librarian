from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.episode_meta import _collect_episode_paths  # noqa: E402

INF = 10**9

_LINE_RE = re.compile(r"^\s*(\d+)\t(.*)$", re.MULTILINE)
_SED_READ_RE = re.compile(r"sed\s+-n\s+['\"]?(\d+),\s*(\d+)p['\"]?\s+(\S+)")
_HEAD_RE = re.compile(r"\bhead\s+(?:-n\s*)?-?(\d+)\s+(\S+)")
_TAIL_RE = re.compile(r"\btail\s+(?:-n\s*)?-?(\d+)\s+(\S+)")
_CAT_RE = re.compile(r"\bcat\s+(\S+)")

_SED_WRITE_RE = re.compile(r"sed\s+-i\s+['\"]?[^'\"\s]+['\"]?\s+(\S+)")
_REDIRECT_RE = re.compile(r"(?<!\d)\s+>\s*([^\s&]+)")
_RM_RE = re.compile(r"\brm\s+(?:-\w+\s+)*(\S+)")
_TEE_RE = re.compile(r"\btee\s+(?:-\w+\s+)*(\S+)")

_DIR_LIST_HEAD = "files and directories up to 2 levels deep"
_BASH_TRUNC = "[Previous command outputs are truncated"
_REJECTED = "REJECTED:"
_NO_OUTPUT_RE = re.compile(r"^\s*\(command exited with code -?\d+\)\s*$")
_EDITOR_ERR_RE = re.compile(r"^\s*(Error|ERROR):", re.MULTILINE)


def _is_failed(obs: str) -> bool:
    head = obs[:300]
    if _REJECTED in head or "Invalid JSON input for tool" in head:
        return True
    return bool(_NO_OUTPUT_RE.match(obs))


def _is_dir_listing(obs: str) -> bool:
    return _DIR_LIST_HEAD in obs[:200]


def _parse_shown_lines(obs: str) -> dict[int, str]:
    out: dict[int, str] = {}
    if "\t" not in obs:
        return out
    for m in _LINE_RE.finditer(obs):
        out[int(m.group(1))] = m.group(2)
    return out


def extract_actions(call: dict) -> list[dict]:
    out: list[dict] = []
    action = call.get("action")
    obs = call.get("observation")
    if not isinstance(action, list):
        return out
    obs_list = obs if isinstance(obs, list) else []

    for idx, a in enumerate(action):
        if not isinstance(a, dict):
            continue
        name = a.get("name", "")
        try:
            args = json.loads(a.get("arguments") or "{}")
        except Exception:
            args = {}
        obs_text = ""
        if idx < len(obs_list) and isinstance(obs_list[idx], dict):
            obs_text = obs_list[idx].get("output", "") or ""

        if name == "bash":
            cmd = args.get("command") or ""
            failed = _is_failed(obs_text)
            if not failed and not _is_dir_listing(obs_text):
                for m in _SED_READ_RE.finditer(cmd):
                    out.append({"kind": "read", "file": m.group(3),
                                "start": int(m.group(1)),
                                "end": int(m.group(2)),
                                "shown": _parse_shown_lines(obs_text),
                                "obs_chars": len(obs_text)})
                for m in _HEAD_RE.finditer(cmd):
                    out.append({"kind": "read", "file": m.group(2),
                                "start": 1, "end": int(m.group(1)),
                                "shown": _parse_shown_lines(obs_text),
                                "obs_chars": len(obs_text)})
                for m in _CAT_RE.finditer(cmd):
                    sh = _parse_shown_lines(obs_text)
                    end = max(sh) if sh else INF
                    out.append({"kind": "read", "file": m.group(1),
                                "start": 1, "end": end,
                                "shown": sh,
                                "obs_chars": len(obs_text)})
            for m in _SED_WRITE_RE.finditer(cmd):
                if not _is_failed(obs_text):
                    out.append({"kind": "write", "file": m.group(1)})
            for m in _REDIRECT_RE.finditer(cmd):
                f = m.group(1)
                if not f.startswith("&"):
                    out.append({"kind": "write", "file": f})
            for m in _RM_RE.finditer(cmd):
                out.append({"kind": "write", "file": m.group(1)})
            for m in _TEE_RE.finditer(cmd):
                out.append({"kind": "write", "file": m.group(1)})

        elif name == "str_replace_editor":
            cmd = args.get("command")
            p = args.get("path") or ""
            if not p:
                continue
            if cmd == "view":
                if _is_failed(obs_text) or _is_dir_listing(obs_text):
                    continue
                vr = args.get("view_range") or []
                if isinstance(vr, str):
                    try: vr = json.loads(vr)
                    except Exception: vr = []
                sh = _parse_shown_lines(obs_text)
                if isinstance(vr, list) and len(vr) == 2 and vr[0] > 0:
                    start, end = int(vr[0]), int(vr[1])
                    if end < 0 or end == -1:
                        end = max(sh) if sh else INF
                else:
                    start = min(sh) if sh else 1
                    end = max(sh) if sh else INF
                out.append({"kind": "read", "file": p,
                            "start": start, "end": end,
                            "shown": sh,
                            "obs_chars": len(obs_text)})
            elif cmd in {"str_replace", "create", "insert"}:
                if _EDITOR_ERR_RE.search(obs_text[:300]):
                    continue
                out.append({"kind": "write", "file": p})
    return out


def _overlap_intervals(history: list[tuple[int, int]],
                       new_start: int, new_end: int
                       ) -> list[tuple[int, int]]:
    out = []
    for (s, e) in history:
        if new_end < s or new_start > e:
            continue
        out.append((max(new_start, s), min(new_end, e)))
    return out


def _overlap_real_tokens(shown: dict[int, str],
                         overlaps: list[tuple[int, int]]) -> int:
    if not shown:
        return 0
    total_chars = 0
    seen_lines: set[int] = set()
    for (s, e) in overlaps:
        for lno in range(s, e + 1):
            if lno in seen_lines:
                continue
            seen_lines.add(lno)
            ln = shown.get(lno)
            if ln is not None:
                total_chars += len(ln) + 1
    return total_chars // 4


def _canonical_aid(aid: str, runner_class: str) -> str:
    if not aid:
        return "?"
    has_persistent_librarian = (
        "Librarian" in runner_class
    )
    if has_persistent_librarian and aid.startswith("librarian_"):
        return "librarian"
    return aid


def analyze_one(path: Path) -> dict:
    try:
        d = json.loads(Path(path).read_text())
    except Exception as e:
        return {"_path": str(path), "_error": f"{type(e).__name__}: {e}"}

    runner_class = d.get("agent") or ""
    calls = d.get("llm_calls") or []
    hist_inter: dict[str, list[tuple[int, int]]] = defaultdict(list)
    hist_intra: dict[str, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: defaultdict(list))

    n_reads = 0
    n_red_inter = 0
    n_red_intra = 0
    real_inter = 0
    real_intra = 0
    out_red_inter = 0
    out_red_intra = 0
    n_writes = 0

    for c in calls:
        aid = _canonical_aid(c.get("agent_id") or "?", runner_class)
        out_tok = int(c.get("output_tokens") or 0)
        actions = extract_actions(c)
        turn_red_inter = False
        turn_red_intra = False

        for a in actions:
            f = a["file"]
            if a["kind"] == "write":
                n_writes += 1
                hist_inter[f] = []
                for aid2 in list(hist_intra.keys()):
                    hist_intra[aid2][f] = []
                continue

            n_reads += 1
            start, end = a["start"], a["end"]

            ov_inter = _overlap_intervals(hist_inter[f], start, end)
            ov_intra = _overlap_intervals(hist_intra[aid][f], start, end)

            if ov_inter:
                n_red_inter += 1
                turn_red_inter = True
                real_inter += _overlap_real_tokens(a["shown"], ov_inter)
            if ov_intra:
                n_red_intra += 1
                turn_red_intra = True
                real_intra += _overlap_real_tokens(a["shown"], ov_intra)

            hist_inter[f].append((start, end))
            hist_intra[aid][f].append((start, end))

        if turn_red_inter:
            out_red_inter += out_tok
        if turn_red_intra:
            out_red_intra += out_tok

    return {
        "_path": str(path),
        "n_reads": n_reads,
        "n_writes": n_writes,
        "n_reads_redundant_inter": n_red_inter,
        "n_reads_redundant_intra": n_red_intra,
        "redundant_real_tokens_inter": real_inter,
        "redundant_real_tokens_intra": real_intra,
        "redundant_output_tokens_inter": out_red_inter,
        "redundant_output_tokens_intra": out_red_intra,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("roots", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL.")
    ap.add_argument("--workers", type=int,
                    default=max(1, (os.cpu_count() or 2) // 2))
    args = ap.parse_args()

    paths = _collect_episode_paths(tuple(args.roots))
    n = 0
    with args.out.open("w") as f:
        if args.workers <= 1:
            for p in paths:
                r = analyze_one(p)
                f.write(json.dumps(r) + "\n"); n += 1
        else:
            with Pool(processes=args.workers) as pool:
                for r in pool.imap_unordered(analyze_one, paths,
                                              chunksize=4):
                    f.write(json.dumps(r) + "\n"); n += 1
    print(f"analyzed {n} episodes (workers={args.workers}) → {args.out}",
          file=sys.stderr)


if __name__ == "__main__":
    main()
