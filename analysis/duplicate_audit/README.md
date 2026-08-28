# Duplicate-output verbatim audit

How much of the output the paper flags as duplicate is *literally* re-emitted
text, and how much is new reasoning wrapped around a redundant re-read?

The turn-level duplicate-read flag counts every output token emitted on a turn
that re-reads an already-seen file range. That count is an **upper bound** on
literally duplicated content. These two scripts measure how loose the bound is.

## Method

Both replicate the duplicate-read flag of `scripts/episode_turns.py` — file
line-range overlap with any earlier read in the same episode, across
invocations, with intervening writes resetting history. For a flagged turn they
then measure the fraction of emitted characters (assistant text including
thinking, plus tool-call arguments) that verbatim-match a line already shown
earlier. Matching is line-level after stripping line-number prefixes and all
whitespace; short lines never match, and paraphrases count as new.

- `dup_upper_bound_audit.py` — the flagged turn itself, minimum line length 20.
- `dup_audit_v2.py` — the turn *consuming* the duplicate observation, i.e. the
  next turn of the same agent, matched only against content first shown
  strictly before the duplicate read was issued, minimum line length 12. Also
  splits out post-`</think>` answer text.

Data is the 100-task subsample (`random.Random(42).sample` over the 500-task
universe) × {HyperAgent, BOAD} × {Qwen3.6-27B, Qwen3.6-35B-A3B}, seed 0, so 400
bare-MAS episodes.

## Result

| Cell | Flagged turns | Flagged share of out tok | Verbatim share, same turn | Verbatim share, consuming turn |
|---|---:|---:|---:|---:|
| 27B / BOAD | 1,304 | 8.5% | 10.6% | 8.4% |
| 27B / HyperAgent | 2,847 | 13.2% | 3.9% | 5.7% |
| 35B-A3B / BOAD | 1,343 | 8.9% | 13.0% | 10.5% |
| 35B-A3B / HyperAgent | 3,563 | 12.9% | 7.8% | 25.6% |
| **pooled** | 9,057 | **11.4%** | **7.7%** | **15.1%** |

Strict verbatim re-quotes are only ~8–15% of the tokens on duplicate-read
turns; the rest is reasoning and summarization surrounding the redundant
exploration. The verbatim measure is itself a lower bound, since paraphrased
re-derivations escape line matching, so true duplication sits between the two.

This is why the paper's causal claim rests on the **intervention** rather than
on the accounting. Removing the re-exploration cuts the replaced role's output
by 55–86% while pass rate stays statistically non-inferior — see
`../significance/`. Had the flagged tokens been mostly new task-relevant
reasoning, removing them could not have left pass rate intact.

## Running

```bash
export EPISODE_LOG_PATH=/data/librarian_logs:$PWD/logs
python analysis/duplicate_audit/dup_upper_bound_audit.py
python analysis/duplicate_audit/dup_audit_v2.py
```
