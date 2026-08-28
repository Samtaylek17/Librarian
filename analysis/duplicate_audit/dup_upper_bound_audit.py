"""Audit: how much of the duplicate-flagged output is verbatim re-emission?

For the same 100-task subsample used in the supplementary experiments
(random.Random(42).sample over the 500-task universe), replicate the
turn-level duplicate-read flag of scripts/episode_turns.py on the bare
MAS archives (HyperAgent / BOAD x 27B / 35B-A3B, seed 0), and for every
duplicate-flagged turn measure the fraction of its emitted characters
(assistant text + tool-call arguments) that verbatim-match a line already
shown earlier in the episode (any prior observation or prior LLM output).

Matching is line-level after normalization (strip leading line-number
prefixes, drop ALL whitespace); lines shorter than 20 normalized chars
never match -> counted as new. Paraphrased re-emissions also count as
new. Both choices are conservative: they can only UNDERSTATE the
verbatim share.
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


def _norm_lines(text):
    out = []
    for ln in text.splitlines():
        ln = _LINENO.sub("", ln)
        ln = _WS.sub("", ln)
        if len(ln) >= 20:
            out.append(ln)
    return out


def _overlap(s, e, ps, pe):
    return s <= pe and e >= ps


def _emitted_text(call):
    parts = [call.get("text") or ""]
    for a in call.get("action") or []:
        if isinstance(a, dict):
            parts.append(a.get("arguments") or "")
    return "\n".join(parts)


def audit_episode(args):
    path, model, agent = args
    try:
        d = json.loads(Path(path).read_text())
    except Exception:
        return None
    runner_class = d.get("agent") or ""
    hist = defaultdict(lambda: defaultdict(list))
    seen = set()
    rows = []  # (out_tok, frac_verbatim) per duplicate-flagged turn
    tot_out = 0
    for c in d.get("llm_calls") or []:
        out_tok = c.get("output_tokens") or 0
        tot_out += out_tok
        aid = _canonical_aid(c.get("agent_id") or "?", runner_class)
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

        emitted = _emitted_text(c)
        if is_dup:
            lines = _norm_lines(emitted)
            n_chars = sum(len(x) for x in lines)
            # short (<20 char) lines: count their mass as new
            short_chars = sum(
                len(_WS.sub("", ln)) for ln in emitted.splitlines()
                if 0 < len(_WS.sub("", _LINENO.sub("", ln))) < 20)
            dup_chars = sum(len(x) for x in lines if x in seen)
            denom = n_chars + short_chars
            frac = dup_chars / denom if denom else 0.0
            rows.append((out_tok, frac))
        # after generation, this turn's output and observations become "seen"
        for ln in _norm_lines(emitted):
            seen.add(ln)
        for o in c.get("observation") or []:
            if isinstance(o, dict):
                for ln in _norm_lines(o.get("output") or ""):
                    seen.add(ln)
    return (model, agent, tot_out, rows)


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

    agg = defaultdict(lambda: {"eps": 0, "tot_out": 0, "flag_tok": 0.0,
                               "verb_tok": 0.0, "n_dup_turns": 0})
    with Pool(12) as pool:
        for res in pool.imap_unordered(audit_episode, jobs, chunksize=4):
            if res is None:
                continue
            model, agent, tot_out, rows = res
            a = agg[(model, agent)]
            a["eps"] += 1
            a["tot_out"] += tot_out
            for out_tok, frac in rows:
                a["flag_tok"] += out_tok
                a["verb_tok"] += out_tok * frac
                a["n_dup_turns"] += 1

    print(f"{'cell':28s} {'eps':>4s} {'dupTurns':>8s} {'flagTok(K)':>10s} "
          f"{'verbTok(K)':>10s} {'verb%':>6s} {'flag%ofOut':>10s}")
    T = {"flag": 0.0, "verb": 0.0, "out": 0}
    for (model, agent), a in sorted(agg.items()):
        pct = 100 * a["verb_tok"] / a["flag_tok"] if a["flag_tok"] else 0
        share = 100 * a["flag_tok"] / a["tot_out"] if a["tot_out"] else 0
        print(f"{model+' / '+agent:28s} {a['eps']:4d} {a['n_dup_turns']:8d} "
              f"{a['flag_tok']/1e3:10.1f} {a['verb_tok']/1e3:10.1f} "
              f"{pct:5.1f}% {share:9.1f}%")
        T["flag"] += a["flag_tok"]; T["verb"] += a["verb_tok"]; T["out"] += a["tot_out"]
    print(f"{'POOLED':28s} {'':4s} {'':8s} {T['flag']/1e3:10.1f} "
          f"{T['verb']/1e3:10.1f} {100*T['verb']/T['flag']:5.1f}% "
          f"{100*T['flag']/T['out']:9.1f}%")


if __name__ == "__main__":
    main()
