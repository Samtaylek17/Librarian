#!/usr/bin/env python
"""Generate self-contained evidence dumps for each Librarian failure case:
issue excerpt + Librarian submit_result answers + gold(base-pass) vs lib(fail)
edited files and their source patch hunks. Run with base anaconda python."""
import json,glob,re,os,sys

_REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_REPO)
from analysis._common import log_dir  # noqa: E402

OUT=os.path.join(os.path.dirname(os.path.abspath(__file__)),"evidence")

def find(d,t,a):
    g=glob.glob(f"{log_dir(d)}/{t}/{a}/SWEBenchEvaluator_*.json"); return g[0] if g else None
def pf(p): return re.findall(r'^\+\+\+ b/(.+)$',p or "",re.M)
def issue_excerpt(J):
    inp=J["llm_calls"][0].get("input")
    if isinstance(inp,list):
        for m in inp:
            c=m.get("content") if isinstance(m,dict) else None
            if isinstance(c,str) and ("<issue>" in c or "pr_description" in c or "PR description" in c):
                i=c.find("<issue>"); j=c.find("</issue>")
                if i!=-1 and j!=-1: return c[i+7:j].strip()
                return c[:1200]
    return ""
def lib_submits(J):
    out=[]
    for t in J["llm_calls"]:
        if "librarian" in t.get("agent_id","").lower():
            obs=t.get("observation")
            if isinstance(obs,list):
                for o in obs:
                    if o.get("tool_name") in ("submit_result","finish","submit"):
                        out.append(o.get("output","").replace("OBSERVATION:\n","").strip())
    return out
def src_hunks(patch):
    # keep only diff blocks for non-test source files, trimmed
    blocks=re.split(r'(?=^diff --git )',patch or "",flags=re.M)
    keep=[]
    for b in blocks:
        m=re.search(r'^\+\+\+ b/(.+)$',b,re.M)
        if not m: continue
        keep.append(b.strip())
    return "\n".join(keep)

# (task, model, config_dir, family, libep, baseep, mode, note)
CASES=[
 ("scikit-learn__scikit-learn-14629","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","definition-anchoring","fix in subclass multioutput.py, lib pointed at cross_val_predict definition"),
 ("scikit-learn__scikit-learn-14629","27B","qwen36_27b_fp8_500_5agents","BOAD","boad_librarian_seed0","boad_seed0","definition-anchoring","same, 27B"),
 ("sympy__sympy-21379","35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","symptom-vs-root","lib -> hyperbolic.py (symptom); gold core/mod.py (root)"),
 ("sympy__sympy-17630","35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","symptom-vs-root","lib blockmatrix.py; gold matexpr.py"),
 ("django__django-13158","35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","adjacent-module","lib sql/query.py; gold sql/compiler.py"),
 ("matplotlib__matplotlib-20826","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","adjacent-module","lib axes/_base.py; gold axis.py"),
 ("pydata__xarray-6938","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","adjacent-module","lib dataset.py; gold variable.py"),
 ("pylint-dev__pylint-7080","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","adjacent-module","lib pylinter.py; gold expand_modules.py"),
 ("sphinx-doc__sphinx-9591","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","adjacent-module","lib ext/autodoc; gold domains/python.py"),
 ("django__django-12193","27B","qwen36_27b_fp8_500_5agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","adjacent-module","lib postgres/forms/array.py; gold forms/widgets.py"),
 ("django__django-11149","35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","stale-negative","lib claimed 'NO tests for view-only m2m'; gold adds that test"),
 ("pydata__xarray-3993","27B","qwen36_27b_fp8_500_5agents","BOAD","boad_librarian_seed0","boad_seed0","stale-negative","lib claimed 'no test uses integrate w/ dim'"),
 # The two search-crowdout cases below are kept for reference only; the mode is
 # not claimed in the paper, and matplotlib-24026 is a get_patch extraction
 # artifact rather than a Librarian failure. See README.md.
 ("matplotlib__matplotlib-24026","35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0","search-crowdout (retracted)","harness patch-extraction artifact, not a Librarian failure"),
 ("django__django-11265","35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0","search-crowdout (not claimed)","16 location dumps; agent edited only the test file"),
]
os.makedirs(OUT,exist_ok=True)
index=[]
for task,model,d,fam,libep,baseep,mode,note in CASES:
    lp=find(d,task,libep); bp=find(d,task,baseep)
    if not lp or not bp:
        print("MISSING",task,model); continue
    dl=json.load(open(lp)); db=json.load(open(bp))
    fn=f"{task}__{model}_{fam}.md"
    with open(os.path.join(OUT,fn),"w") as f:
        f.write(f"# {task}  ({model} / {fam})\n\n")
        f.write(f"**Failure mode:** {mode}  \n**Note:** {note}\n\n")
        f.write(f"- base (no Librarian) `{baseep}`: passed={db['passed']}\n")
        f.write(f"- +Librarian `{libep}`: passed={dl['passed']}\n\n")
        f.write(f"**Gold/base edited files:** {pf(db.get('patch'))}\n\n")
        f.write(f"**+Librarian edited files:** {pf(dl.get('patch'))}\n\n")
        f.write("## Issue\n\n```\n"+issue_excerpt(dl)[:1600]+"\n```\n\n")
        f.write("## Librarian submit_result answers (the search guidance)\n\n")
        for i,s in enumerate(lib_submits(dl),1):
            f.write(f"{i}. {s[:700]}\n\n")
        f.write("## base (PASS) patch\n\n```diff\n"+src_hunks(db.get('patch'))[:3500]+"\n```\n\n")
        f.write("## +Librarian (FAIL) patch\n\n```diff\n"+(src_hunks(dl.get('patch'))[:3500] or "(empty)")+"\n```\n")
    index.append((mode,task,model,fam,fn))
    print("wrote",fn)
# index
with open(os.path.join(OUT,"INDEX.md"),"w") as f:
    f.write("# Evidence index\n\n| Mode | Task | Model | Family | File |\n|---|---|---|---|---|\n")
    for mode,task,model,fam,fn in sorted(index):
        f.write(f"| {mode} | {task} | {model} | {fam} | [{fn}]({fn}) |\n")
print("wrote INDEX.md")
