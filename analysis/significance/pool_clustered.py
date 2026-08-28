"""Task-clustered pooled NI / equivalence test at n=500.

For each comparison (base->+Lib, +Cav->+Cav+Lib), average each task's paired
pass differences over the four (model, family) settings, then run one-sample
tests over the n=500 independent tasks.
"""
import json
import os
import sys
from collections import defaultdict
from math import sqrt

from scipy import stats

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    _REPO, "analysis", "data", "table45_episodes.jsonl")

recs = defaultdict(dict)  # (model, family, task) -> {method: passed}
with open(EP) as f:
    for line in f:
        r = json.loads(line)
        if not r.get("valid"):
            continue
        recs[(r["model"], r["family"], r["task_id"])][r["method"]] = bool(r["passed"])

COMPARISONS = {
    "A base->+Lib": [("HyperAgent", "+Librarian"), ("BOAD", "+Librarian")],
    "B +Cav->+Cav+Lib": [("+Caveman", "+Caveman+Librarian"), ("+Caveman", "+Caveman+Librarian")],
}


def run(name, base_of_family, var_method):
    per_task = defaultdict(list)
    for (model, family, task), methods in recs.items():
        base_m = base_of_family(family)
        if base_m in methods and var_method in methods:
            per_task[task].append(int(methods[var_method]) - int(methods[base_m]))
    d = [sum(v) / len(v) for v in per_task.values()]
    n = len(d)
    mean = sum(d) / n
    sd = sqrt(sum((x - mean) ** 2 for x in d) / (n - 1))
    se = sd / sqrt(n)
    # one-sided NI at 2pp margin: H1 mean > -0.02
    t_ni = (mean + 0.02) / se
    p_ni = stats.t.sf(t_ni, df=n - 1)
    # 90% CI -> achieved equivalence bound (TOST at alpha=0.05)
    tcrit = stats.t.ppf(0.95, df=n - 1)
    lo, hi = mean - tcrit * se, mean + tcrit * se
    eq_bound = max(abs(lo), abs(hi))
    # achieved NI margin (smallest margin still significant one-sided)
    ach_ni = max(0.0, tcrit * se - mean)
    # settings per task distribution
    from collections import Counter
    cnt = Counter(len(v) for v in per_task.values())
    print(f"{name}: n_tasks={n} settings/task={dict(cnt)}")
    print(f"  mean={mean*100:+.2f}pp  se={se*100:.2f}pp  NI2pp t={t_ni:.2f} p={p_ni:.2e}")
    print(f"  90%CI=[{lo*100:+.2f},{hi*100:+.2f}]pp  TOST eq bound=±{eq_bound*100:.2f}pp  achieved NI margin={ach_ni*100:.2f}pp")


run("A base->+Lib", lambda fam: fam if fam in ("HyperAgent", "BOAD") else None, "+Librarian")


def base_cav(_fam):
    return "+Caveman"


run("B +Cav->+Cav+Lib", base_cav, "+Caveman+Librarian")
