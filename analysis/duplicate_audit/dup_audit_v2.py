"""Variant B: verbatim share of the turn CONSUMING a duplicate-read obs.

For each turn whose previous same-agent turn issued a duplicate file read,
measure how much of its emitted text (full, and post-</think> answer+args
only) verbatim-matches lines first shown STRICTLY BEFORE that duplicate
read was issued (i.e. content the episode already had; the re-read added
nothing).
"""
import json
import random
import re
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
from analysis._common import log_dir  # noqa: E402
from analysis.redundancy_analysis import extract_actions, _canonical_aid  # noqa: E402

ROOTS = {
    "27B": Path(log_dir("qwen36_27b_fp8_500_5agents")),
    "35B-A3B": Path(log_dir("qwen36_35b_a3b_500_3agents")),
}
AGENTS = {"HyperAgent": "hyperagent_seed0", "BOAD": "boad_seed0"}

_LINENO = re.compile(r"^\s*\d+[\t |:]")
_WS = re.compile(r"\s+")
MINLEN = 12


def _norm_lines(text):
    out = []
    for ln in text.splitlines():
        ln = _WS.sub("", _LINENO.sub("", ln))
        if len(ln) >= MINLEN:
            out.append(ln)
    return out


def _overlap(s, e, ps, pe):
    return s <= pe and e >= ps


def _emitted(call, answer_only=False):
    t = call.get("text") or ""
    if answer_only and "</think>" in t:
        t = t.split("</think>", 1)[1]
    parts = [t]
    for a in call.get("action") or []:
        if isinstance(a, dict):
            parts.append(a.get("arguments") or "")
    return "\n".join(parts)


def _frac(text, seen, before_idx):
    lines = _norm_lines(text)
    tot = sum(len(x) for x in lines)
    dup = sum(len(x) for x in lines if seen.get(x, 1 << 60) < before_idx)
    return dup, tot


def audit_episode(args):
    path, model, agent = args
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return None
    runner_class = d.get("agent") or ""
    hist = defaultdict(lambda: defaultdict(list))
    seen = {}  # norm line -> first turn idx shown
    last_dup_read_turn = {}  # agent id -> turn idx of its last turn, if dup
    rows = []
    for ti, c in enumerate(d.get("llm_calls") or []):
        out_tok = c.get("output_tokens") or 0
        aid = _canonical_aid(c.get("agent_id") or "?", runner_class)

        # is this turn consuming a duplicate observation?
        prev = last_dup_read_turn.get(aid)
        if prev is not None:
            dup_f, tot_f = _frac(_emitted(c), seen, prev)
            dup_a, tot_a = _frac(_emitted(c, answer_only=True), seen, prev)
            rows.append((out_tok, dup_f, tot_f, dup_a, tot_a))

        is_dup = False
        for a in extract_actions(c):
            f = a["file"]
            if a["kind"] == "write":
                for aid2 in list(hist.keys()):
                    hist[aid2][f] = []
                continue
            s, e = a["start"], a["end"]
            if not is_dup:
                dup = any(_overlap(s, e, ps, pe) for ps, pe in hist[aid][f])
                if not dup:
                    dup = any(
                        _overlap(s, e, ps, pe)
                        for aid2, files in hist.items() if aid2 != aid
                        for ps, pe in files.get(f, []))
                if dup:
                    is_dup = True
            hist[aid][f].append((s, e))
        last_dup_read_turn[aid] = ti if is_dup else None

        for ln in _norm_lines(_emitted(c)):
            seen.setdefault(ln, ti)
        for o in c.get("observation") or []:
            if isinstance(o, dict):
                for ln in _norm_lines(o.get("output") or ""):
                    seen.setdefault(ln, ti)
    return (model, agent, rows)


def main():
    jobs = []
    for model, root in ROOTS.items():
        ids = sorted(p.name for p in root.iterdir() if p.is_dir())
        sample = random.Random(42).sample(ids, 100)
        for agent, sub in AGENTS.items():
            for tid in sample:
                g = list((root / tid / sub).glob("SWEBenchEvaluator_*.json"))
                if g:
                    jobs.append((str(g[0]), model, agent))
    print(f"{len(jobs)} episodes", file=sys.stderr)

    agg = defaultdict(lambda: [0, 0.0, 0.0, 0.0, 0.0])  # n, dupF, totF, dupA, totA
    with Pool(12) as pool:
        for res in pool.imap_unordered(audit_episode, jobs, chunksize=4):
            if res is None:
                continue
            model, agent, rows = res
            a = agg[(model, agent)]
            for out_tok, dF, tF, dA, tA in rows:
                a[0] += 1; a[1] += dF; a[2] += tF; a[3] += dA; a[4] += tA
    print(f"{'cell':28s} {'consTurns':>9s} {'full-verb%':>10s} {'ans-verb%':>9s}")
    TF = [0.0, 0.0, 0.0, 0.0]
    for (model, agent), a in sorted(agg.items()):
        print(f"{model+' / '+agent:28s} {a[0]:9d} "
              f"{100*a[1]/a[2] if a[2] else 0:9.1f}% "
              f"{100*a[3]/a[4] if a[4] else 0:8.1f}%")
        TF[0] += a[1]; TF[1] += a[2]; TF[2] += a[3]; TF[3] += a[4]
    print(f"{'POOLED':28s} {'':9s} {100*TF[0]/TF[1]:9.1f}% {100*TF[2]/TF[3]:8.1f}%")


if __name__ == "__main__":
    main()
