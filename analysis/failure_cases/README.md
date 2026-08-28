# Librarian failure cases

Where does the Librarian *hurt*? This directory covers every SWE-bench task
where the bare MAS backbone passed but adding the Librarian made it fail,
pooled over the four `base → +Librarian` cells (2 backbones × {HyperAgent,
BOAD}, seed 0, 500 tasks each) — **112 losing episodes** — and drills into the
ones actually attributable to the Librarian's guidance.

## Localization split

For each of the 112 losses, `localize.py` compares the source files the
Librarian-guided agent edited against the files the *successful* base run
edited, a per-task proxy for correct localization.

| Category | n | % | Meaning |
|---|---:|---:|---|
| same | 62 | 55% | Librarian localized to the exact gold file; the regression is in the edit, not the search |
| partial | 29 | 26% | Overlapping but not identical file set |
| disjoint | 11 | 10% | Steered to a completely different source file, i.e. mis-localization |
| empty | 10 | 9% | No net source edit survived |

The majority of Librarian losses localize to the exact file the winning run
edited, so they are downstream editing errors shared with the base MAS rather
than search failures. Librarian-attributable losses are **13** — the 11
disjoint cases plus 2 overconfident-negative cases that sit in the same/partial
buckets by file overlap.

## Failure modes

1. **Symbol-definition anchoring** (11 cases). The Librarian answers the
   literal "where is `X`?" with the definition site while the fix lives in a
   caller, subclass, or root-cause module it never surfaces. In
   `scikit-learn-14629` it correctly locates `_enforce_prediction_order` in
   `_validation.py`, but the gold fix adds a `classes_` property to
   `MultiOutputClassifier` in `multioutput.py`. Also `sympy-21379` (symptom
   module `hyperbolic.py` versus root cause `core/mod.py`), `sympy-17630`,
   `django-13158`, `matplotlib-20826`, `xarray-6938`, `pylint-7080`,
   `sphinx-9591`, `django-12193`.
2. **Overconfident negative guidance** (2 cases). A confident "none exists"
   closes off required work. In `django-11149` the Librarian answers
   "Confirmed: there are NO tests for view-only permissions on ManyToMany
   inlines". True as stated, but writing that missing test is part of the
   required fix.

The common mechanism is that the Librarian resolves the *named symbol*
faithfully but does not reason about *which neighbor* of that symbol is the fix
site. The targeted design fix is to have it return the caller / root-cause set
rather than only the definition.

A third candidate mode, search round-trips crowding out the edit, was dropped.
Its clearest example, `matplotlib-24026`, turned out to be a harness artifact —
the agent did produce and commit a plausible fix to `stackplot.py`, and the
recorded empty patch came from a `get_patch` extraction bug that has since been
fixed. The `empty` bucket above may contain further such artifacts; re-audit
before quoting its count.

## Contents

```
localize.py       regenerates evidence/localization_split.txt (the 112-loss split)
gen_evidence.py   regenerates evidence/<task>__<model>_<family>.md
evidence/
  INDEX.md        table of the case dumps
  <task>*.md      per case: issue excerpt, the Librarian's submit_result
                  answers, gold(PASS) patch vs +Librarian(FAIL) patch
  localization_split.txt
```

Raw episodes are 30–50 MB each and are not copied here. `evidence/` is the
self-contained record; the scripts regenerate it from the logs.

## Running

```bash
export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs
python analysis/failure_cases/localize.py       # needs analysis/data/table45_episodes.jsonl
python analysis/failure_cases/gen_evidence.py
```

`localize.py` reads the episode records built by
`analysis/significance/extract_table45.py`.
