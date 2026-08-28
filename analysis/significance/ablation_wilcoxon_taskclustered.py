"""Per-mechanism task-clustered test: average each task's (HA, BOAD) paired diffs -> n=100 Wilcoxon."""
import os
import sys
from pathlib import Path

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "scripts"))
from analysis._common import log_dir  # noqa: E402
from episode_meta import extract_one  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

ABL = Path(log_dir("librarian_ablation_100"))
BASE = Path(log_dir("qwen36_27b_fp8_500_5agents"))
HOSTS = {"hyperagent": "hyperagent_librarian_seed0", "boad": "boad_librarian_seed0"}

def meta_for(d):
    if not d.is_dir(): return None
    fs = sorted(d.glob("SWEBenchEvaluator_*.json"))
    if not fs: return None
    r = extract_one(fs[-1])
    if r is None or r.get("eval_error") or not r.get("n_llm_calls"): return None
    return r

tasks = sorted(d.name for d in ABL.iterdir() if d.is_dir())
base_cache = {(h, t): meta_for(BASE / t / bs) for h, bs in HOSTS.items() for t in tasks}

for cond in ["nopersist", "nopointer", "noscope", "nofresh"]:
    d_out, d_j = [], []
    for t in tasks:
        outs, js = [], []
        for h in HOSTS:
            b = base_cache[(h, t)]
            a = meta_for(ABL / t / f"{h}_librarian_{cond}_seed0")
            if b is None or a is None: continue
            if b.get("output_tokens") is None or a.get("output_tokens") is None: continue
            outs.append(a["output_tokens"] - b["output_tokens"])
            if b.get("total_joules") and a.get("total_joules"):
                js.append(a["total_joules"] - b["total_joules"])
        if outs: d_out.append(sum(outs)/len(outs))
        if js: d_j.append(sum(js)/len(js))
    w_out, w_j = wilcoxon(d_out), wilcoxon(d_j)
    print(f"{cond:10s} n={len(d_out)} mean_dout={sum(d_out)/len(d_out)/1000:+6.2f}K p={w_out.pvalue:.2e} | mean_dJ={sum(d_j)/len(d_j)/1000:+6.2f}kJ p={w_j.pvalue:.2e}")
