#!/usr/bin/env python
"""Paired significance tests for Table 4/5 (Librarian / Caveman / LastN).

Per comparison (variant vs family base), paired by task_id over the 500-task
set, keeping only tasks valid in BOTH configs:

  Energy & Output-tokens : mean(base - variant) with paired bootstrap 95% CI
                           + Wilcoxon signed-rank (two-sided).
  Pass rate              : McNemar exact (improvement) AND
                           paired non-inferiority Wald test, margin 2pp (preserve).
  Multiple comparison    : Holm within each metric family.
"""
import sys, json
from collections import defaultdict
import numpy as np
from scipy import stats

import os

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_DEFAULT_INP = os.path.join(_REPO, "analysis", "data", "table45_episodes.jsonl")
INP = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT_INP
NI_MARGIN = 0.02          # non-inferiority margin: 2 percentage points
N_BOOT = 10000
RNG = np.random.default_rng(20260709)

ENERGY_FIELD = "joules"   # table appears to use raw total_joules

recs = [json.loads(l) for l in open(INP)]

# index: (model, family, method) -> {task_id: rec}
by_cfg = defaultdict(dict)
for r in recs:
    by_cfg[(r["model"], r["family"], r["method"])][r["task_id"]] = r

MODELS = ["35B-A3B", "27B"]
FAMS = ["HyperAgent", "BOAD"]
# (variant_label, base_label) comparisons
COMPARE = [
    ("+Librarian", None),           # None -> family base
    ("+LastNObservation", None),
    ("+Caveman", None),
    ("+Caveman+Librarian", None),
    ("+Caveman+Librarian", "+Caveman"),  # isolated librarian on top of caveman
]

def paired(model, family, var, base_label):
    base = base_label or family
    A = by_cfg.get((model, family, base), {})
    B = by_cfg.get((model, family, var), {})
    tasks = [t for t in A if t in B and A[t]["valid"] and B[t]["valid"]]
    return base, tasks, A, B

def boot_ci_mean(diffs, n=N_BOOT):
    d = np.asarray(diffs, float)
    idx = RNG.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))

def metric_test(base_vals, var_vals):
    """base - variant ; positive => variant reduces the metric."""
    b = np.asarray(base_vals, float); v = np.asarray(var_vals, float)
    diff = b - v
    lo, hi = boot_ci_mean(diff)
    # wilcoxon on paired values (drops zero-diffs by default)
    try:
        w_p = float(stats.wilcoxon(b, v, zero_method="wilcox").pvalue)
    except ValueError:
        w_p = float("nan")
    mean_b, mean_v = float(b.mean()), float(v.mean())
    return {
        "mean_base": mean_b, "mean_var": mean_v,
        "mean_reduction": float(diff.mean()),
        "pct_reduction": 100.0 * diff.mean() / mean_b if mean_b else float("nan"),
        "ci_lo": lo, "ci_hi": hi,
        "median_reduction": float(np.median(diff)),
        "wilcoxon_p": w_p,
        "sig_ci": (lo > 0) or (hi < 0),  # CI excludes 0
    }

def pass_test(base_pass, var_pass):
    b = np.asarray(base_pass, bool); v = np.asarray(var_pass, bool)
    N = len(b)
    n01 = int(np.sum(~b & v))   # base fail, variant pass  (variant better)
    n10 = int(np.sum(b & ~v))   # base pass, variant fail  (variant worse)
    disc = n01 + n10
    # McNemar exact (two-sided binomial on discordant pairs)
    if disc > 0:
        mcn_p = float(stats.binomtest(min(n01, n10), disc, 0.5,
                                      alternative="two-sided").pvalue)
    else:
        mcn_p = 1.0
    delta = (n01 - n10) / N          # p_var - p_base
    # paired-proportion Wald variance
    var = (n01 + n10 - (n01 - n10) ** 2 / N) / (N ** 2)
    se = var ** 0.5
    # non-inferiority: H0 delta <= -margin vs H1 delta > -margin
    if se > 0:
        z = (delta + NI_MARGIN) / se
        ni_p = float(stats.norm.sf(z))
    else:
        z = float("inf") if delta > -NI_MARGIN else float("-inf")
        ni_p = 0.0 if delta > -NI_MARGIN else 1.0
    # bootstrap CI of pass-rate diff (pp)
    pair = np.column_stack([b.astype(float), v.astype(float)])
    idx = RNG.integers(0, N, size=(N_BOOT, N))
    ds = (pair[idx, 1].mean(axis=1) - pair[idx, 0].mean(axis=1))
    ci_lo, ci_hi = float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))
    ci_lo5 = float(np.percentile(ds, 5))   # one-sided 95% lower bound (matches NI alpha)
    return {
        "N": N, "pass_base": float(b.mean()), "pass_var": float(v.mean()),
        "diff_pp": 100 * delta, "n01": n01, "n10": n10,
        "mcnemar_p": mcn_p, "ni_z": z, "ni_p": ni_p,
        "ci_lo_pp": 100 * ci_lo, "ci_hi_pp": 100 * ci_hi, "ci_lo5_pp": 100 * ci_lo5,
        "improved_sig": (mcn_p < 0.05 and delta > 0),
        "regressed_sig": (mcn_p < 0.05 and delta < 0),
    }

def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals); adj = [0.0] * m; running = 0.0
    for rank, i in enumerate(order):
        a = min(1.0, (m - rank) * pvals[i])
        running = max(running, a)
        adj[i] = running
    return adj

# ---- run all comparisons ----
rows = []
for model in MODELS:
    for family in FAMS:
        for var, base_label in COMPARE:
            base, tasks, A, B = paired(model, family, var, base_label)
            if len(tasks) < 5:
                continue
            en = metric_test([A[t][ENERGY_FIELD] for t in tasks],
                             [B[t][ENERGY_FIELD] for t in tasks])
            ot = metric_test([A[t]["out_tok"] for t in tasks],
                             [B[t]["out_tok"] for t in tasks])
            pw = pass_test([A[t]["passed"] for t in tasks],
                           [B[t]["passed"] for t in tasks])
            rows.append({"model": model, "family": family, "base": base,
                         "var": var, "n": len(tasks),
                         "energy": en, "outtok": ot, "pass": pw})

# Holm within metric (over all comparisons)
for metric, key, dst in [("energy", "wilcoxon_p", "holm_p"),
                         ("outtok", "wilcoxon_p", "holm_p"),
                         ("pass", "mcnemar_p", "holm_p"),
                         ("pass", "ni_p", "ni_holm")]:
    ps = [r[metric][key] for r in rows]
    adj = holm(ps)
    for r, a in zip(rows, adj):
        r[metric][dst] = a

def verdict(p):
    """Consistent one-sided alpha=0.05 non-inferiority (Holm-adjusted)."""
    if p["improved_sig"]:
        return "IMPROVED"
    if p["regressed_sig"]:
        return "REGRESSED"
    if p["ni_holm"] < 0.05:
        return "NON-INFERIOR"
    return "INCONCLUSIVE"

json.dump(rows, open("table45_stats.json", "w"), indent=2)

# ---- verification vs printed table ----
print("=" * 100)
print("VERIFICATION  (computed means over valid paired-universe of each config, standalone)")
print("=" * 100)
print(f"{'model':8} {'method':22} {'n_valid':>7} {'pass%':>6} {'outtok/K':>9} "
      f"{'joules/K':>9} {'clean/K':>9}")
seen = set()
for model in MODELS:
    for family in FAMS:
        labels = [family, "+Librarian", "+LastNObservation", "+Caveman", "+Caveman+Librarian"]
        for lab in labels:
            d = by_cfg.get((model, family, lab), {})
            v = [r for r in d.values() if r["valid"]]
            if not v: continue
            pa = 100 * np.mean([r["passed"] for r in v])
            ot = np.mean([r["out_tok"] for r in v]) / 1000
            jj = np.mean([r["joules"] for r in v]) / 1000
            jc = np.mean([r["joules_clean"] for r in v if r["joules_clean"] is not None] or [np.nan]) / 1000
            print(f"{model:8} {lab:22} {len(v):7d} {pa:6.1f} {ot:9.1f} {jj:9.2f} {jc:9.2f}")

def fmt_p(p):
    if p != p: return "  nan"
    if p < 1e-3: return f"{p:.1e}"
    return f"{p:.3f}"

print("\n" + "=" * 100)
print("ENERGY  &  OUTPUT-TOKENS  reduction  (base - variant, paired)")
print("=" * 100)
hdr = (f"{'model':8} {'family':10} {'comparison':28} {'n':>4} "
       f"{'red%':>6} {'boot95CI(reduction, native)':>30} {'wilcox_p':>9} {'holm_p':>8} sig")
for metric, unit, div in [("energy", "kJ", 1000), ("outtok", "Ktok", 1000)]:
    print(f"\n--- {metric.upper()} ---")
    print(hdr)
    for r in rows:
        m = r[metric]
        comp = f"{r['var']} vs {r['base']}"
        ci = f"[{m['ci_lo']/div:+.2f},{m['ci_hi']/div:+.2f}]{unit}"
        sig = "***" if m["holm_p"] < 0.05 else ("*" if m["wilcoxon_p"] < 0.05 else "")
        print(f"{r['model']:8} {r['family']:10} {comp:28} {r['n']:4d} "
              f"{m['pct_reduction']:6.1f} {ci:>30} {fmt_p(m['wilcoxon_p']):>9} "
              f"{fmt_p(m['holm_p']):>8} {sig}")

print("\n" + "=" * 100)
print(f"PASS RATE   (McNemar improvement  +  non-inferiority TOST, margin={NI_MARGIN*100:.0f}pp)")
print("=" * 100)
print(f"{'model':8} {'family':10} {'comparison':28} {'n':>4} {'base%':>6} {'var%':>6} "
      f"{'Δpp':>6} {'n01/n10':>8} {'McN_holm':>8} {'NI_holm':>8} {'NI_lo(pp)':>9} verdict")
for r in rows:
    p = r["pass"]
    comp = f"{r['var']} vs {r['base']}"
    print(f"{r['model']:8} {r['family']:10} {comp:28} {p['N']:4d} "
          f"{100*p['pass_base']:6.1f} {100*p['pass_var']:6.1f} {p['diff_pp']:+6.1f} "
          f"{str(p['n01'])+'/'+str(p['n10']):>8} "
          f"{fmt_p(p['holm_p']):>8} {fmt_p(p['ni_holm']):>8} "
          f"{p['ci_lo5_pp']:>+9.1f} {verdict(p)}")
print("\nLegend: McN_holm = McNemar exact two-sided, Holm-adj (improvement/regression).")
print(f"        NI_holm  = paired non-inferiority (margin {NI_MARGIN*100:.0f}pp, 1-sided), Holm-adj.")
print("        NI_lo    = one-sided 95% bootstrap lower bound of Δ=var-base (pp);")
print(f"                   NON-INFERIOR requires it (and NI_holm<.05) above -{NI_MARGIN*100:.0f}pp.")
print("\nwrote table45_stats.json")
