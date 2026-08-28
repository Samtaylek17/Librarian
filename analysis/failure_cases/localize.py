import json,glob,re,os,sys

_REPO=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0,_REPO)
from analysis._common import log_dir  # noqa: E402

def find(dirn,task,ad):
    g=glob.glob(f"{log_dir(dirn)}/{task}/{ad}/SWEBenchEvaluator_*.json"); return g[0] if g else None
def srcfiles(patch):
    fs=re.findall(r'^\+\+\+ b/(.+)$',patch or "",re.M)
    # non-test source files only (fix site)
    return set(f for f in fs if "/tests/" not in f and not os.path.basename(f).startswith("test_")
               and not f.endswith("_test.py"))
def allfiles(patch):
    return set(re.findall(r'^\+\+\+ b/(.+)$',patch or "",re.M))
# cells: base->+Librarian
CELLS=[("35B-A3B","qwen36_35b_a3b_500_3agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0"),
       ("35B-A3B","qwen36_35b_a3b_500_3agents","BOAD","boad_librarian_seed0","boad_seed0"),
       ("27B","qwen36_27b_fp8_500_5agents","HyperAgent","hyperagent_librarian_seed0","hyperagent_seed0"),
       ("27B","qwen36_27b_fp8_500_5agents","BOAD","boad_librarian_seed0","boad_seed0")]
JSONL=sys.argv[1] if len(sys.argv)>1 else os.path.join(
    _REPO,"analysis","data","table45_episodes.jsonl")
recs=[json.loads(l) for l in open(JSONL)]
from collections import defaultdict
by=defaultdict(dict)
for r in recs: by[(r["model"],r["family"],r["method"])][r["task_id"]]=r

tot=dict(same=0,disjoint=0,partial=0,empty=0,n=0)
detail=[]
for model,d,fam,libep,baseep in CELLS:
    A=by[(model,fam,fam)]; B=by[(model,fam,"+Librarian")]
    lose=[t for t in A if t in B and A[t]["valid"] and B[t]["valid"] and A[t]["passed"] and not B[t]["passed"]]
    for t in lose:
        lp=find(d,t,libep); bp=find(d,t,baseep)
        if not lp or not bp: continue
        dl=json.load(open(lp)); db=json.load(open(bp))
        gs=srcfiles(db.get("patch")); ls=srcfiles(dl.get("patch"))
        tot["n"]+=1
        if not allfiles(dl.get("patch")): cat="empty"
        elif not ls: cat="empty"   # only edited tests
        elif ls==gs: cat="same"
        elif ls & gs: cat="partial"
        else: cat="disjoint"
        tot[cat]=tot.get(cat,0)+1
        detail.append((model,fam,t,cat,sorted(gs),sorted(ls)))
print("Across base->+Librarian LOSSES (base pass, lib fail):")
print(f"  n={tot['n']}")
for k in ["same","partial","disjoint","empty"]:
    print(f"  {k:9}: {tot.get(k,0)}")
print("\n  same     = lib edited exactly the gold source file(s) -> localization OK, edit wrong")
print("  partial  = overlap but not identical")
print("  disjoint = lib edited a completely different source file -> MIS-LOCALIZED")
print("  empty    = lib produced no source edit (only tests / nothing)")
print("\n--- DISJOINT (mis-localized) cases ---")
for m,f,t,c,gs,ls in detail:
    if c=="disjoint":
        print(f"  {m}/{f} {t}\n       gold:{gs}\n        lib:{ls}")
print("\n--- EMPTY (no source edit) cases ---")
for m,f,t,c,gs,ls in detail:
    if c=="empty":
        print(f"  {m}/{f} {t}  gold:{gs} lib:{ls}")
