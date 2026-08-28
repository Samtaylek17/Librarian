#!/usr/bin/env python
"""Pooled (stratified) non-inferiority for the Librarian's pass-rate effect,
plus per-cell achieved one-sided 95% margin. Answers the
'preserve' concern with a properly-powered estimate."""
import json
import os
import sys
from collections import defaultdict
import numpy as np
from scipy import stats

RNG = np.random.default_rng(20260709)
N_BOOT = 20000
_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _REPO, "analysis", "data", "table45_episodes.jsonl")
recs = [json.loads(l) for l in open(INP)]
by_cfg = defaultdict(dict)
for r in recs:
    by_cfg[(r["model"], r["family"], r["method"])][r["task_id"]] = r

def pairs(model, family, base, var):
    A = by_cfg[(model, family, base)]; B = by_cfg[(model, family, var)]
    ts = [t for t in A if t in B and A[t]["valid"] and B[t]["valid"]]
    return np.array([A[t]["passed"] for t in ts]), np.array([B[t]["passed"] for t in ts])

def analyse(label, blocks):
    """blocks: list of (base_arr, var_arr). Pool as independent paired blocks."""
    b = np.concatenate([x[0] for x in blocks]).astype(bool)
    v = np.concatenate([x[1] for x in blocks]).astype(bool)
    N = len(b)
    n01 = int(np.sum(~b & v)); n10 = int(np.sum(b & ~v)); disc = n01 + n10
    mcn = stats.binomtest(min(n01, n10), disc, 0.5).pvalue if disc else 1.0
    delta = (n01 - n10) / N
    se = ((n01 + n10 - (n01 - n10) ** 2 / N) / N ** 2) ** 0.5
    # bootstrap CI of paired pass-rate difference (resample task pairs)
    bf, vf = b.astype(float), v.astype(float)
    idx = RNG.integers(0, N, size=(N_BOOT, N))
    ds = vf[idx].mean(axis=1) - bf[idx].mean(axis=1)
    lo2, hi2 = np.percentile(ds, [2.5, 97.5])
    lo5 = np.percentile(ds, 5)
    # achieved margin (one-sided 95%): most it could hurt
    achieved = -lo5 * 100
    ni2 = 1 - stats.norm.cdf((delta + 0.02) / se) if se > 0 else 0.0
    print(f"{label:42} N={N:4d}  base={100*b.mean():5.1f}  var={100*v.mean():5.1f}  "
          f"Δ={100*delta:+5.2f}pp  95%CI[{100*lo2:+.2f},{100*hi2:+.2f}]  "
          f"McN_p={mcn:.3f}  NI2pp_p={ni2:.4f}  achieved_margin={achieved:.2f}pp")
    return dict(N=N, delta=100*delta, ci=[100*lo2, 100*hi2], mcn=mcn,
                ni2=ni2, achieved=achieved)

MODELS = ["35B-A3B", "27B"]; FAMS = ["HyperAgent", "BOAD"]

print("="*120)
print("A. Librarian ADDED TO BASE MAS  (base -> +Librarian), per cell + pooled")
print("="*120)
blocks = []
for m in MODELS:
    for f in FAMS:
        base = f
        bl = pairs(m, f, base, "+Librarian")
        analyse(f"  {m}/{f}: {base} -> +Librarian", [bl]); blocks.append(bl)
analyse(">> POOLED (4 settings, ~2000 pairs)", blocks)

print("\n" + "="*120)
print("B. Librarian ADDED ON TOP OF CAVEMAN  (+Caveman -> +Caveman+Librarian), per cell + pooled")
print("="*120)
blocks = []
for m in MODELS:
    for f in FAMS:
        bl = pairs(m, f, "+Caveman", "+Caveman+Librarian")
        analyse(f"  {m}/{f}: +Caveman -> +Caveman+Librarian", [bl]); blocks.append(bl)
analyse(">> POOLED (4 settings)", blocks)

print("\n" + "="*120)
print("C. Caveman ADDED TO BASE MAS  (base -> +Caveman), per cell + pooled")
print("="*120)
blocks = []
for m in MODELS:
    for f in FAMS:
        bl = pairs(m, f, f, "+Caveman")
        analyse(f"  {m}/{f}: {f} -> +Caveman", [bl]); blocks.append(bl)
analyse(">> POOLED (4 settings)", blocks)
print("\n" + "="*120)
print("D. FULL EFFICIENCY CONFIG vs BASE  (base -> +Caveman+Librarian), per cell + pooled")
print("="*120)
blocks = []
for m in MODELS:
    for f in FAMS:
        bl = pairs(m, f, f, "+Caveman+Librarian")
        analyse(f"  {m}/{f}: {f} -> +Caveman+Librarian", [bl]); blocks.append(bl)
analyse(">> POOLED (4 settings)", blocks)

print("\nNote: pooling treats each (task,setting) as an independent paired block. Task ids")
print("recur across model/family settings, so the pool has mild positive dependence ->")
print("the pooled CI is marginally optimistic; the per-cell rows are fully independent.")
