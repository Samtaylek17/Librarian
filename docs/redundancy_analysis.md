# File-Read Redundancy Analysis: BOAD and HyperAgent (HA) Families

This document captures the per-episode file-read redundancy decomposition for the
two main MAS architectures we evaluate (BOAD, HyperAgent) and their `+lib`
variants, on the 27B FP8 and 35B-A3B Qwen3.6 models.

## Definitions

### Overlap classification (per intra-redundant read pair)
For a current read `[c, d]` matched to its most-recent overlapping prior
read `[a, b]` by the same canonical agent, classify as:

| Pattern        | Condition                              | Meaning                                 |
|---------------|----------------------------------------|-----------------------------------------|
| `exact`       | `c == a` and `d == b`                  | same range                               |
| `subset`      | `a <= c` and `d <= b` (not exact)      | new fully inside prior (no new info)    |
| `superset`    | `c <= a` and `b <= d` (not exact)      | new contains prior (broader)            |
| `sliding_fwd` | `c > a` and `d > b`                    | shifted forward (chunk-scan)            |
| `sliding_bwd` | `c < a` and `d < b`                    | shifted backward                         |
| `other`       | impossible under overlap precondition  | —                                        |

### Redundancy definitions
- **any-overlap**: any overlap counts as redundant (default).
- **strict containment**: only `subset + exact` count as redundant (new read
  brings 0 new lines).

### Scopes (per-turn output token accounting)
Each LLM turn classified into one of:
- `intra_only`: same canonical agent saw overlapping prior on this file
- `cross_only`: only a different agent saw overlapping prior
- `both`: both kinds of overlap

Aggregates: `intra = intra_only + both`, `cross = cross_only + both`,
`inter = intra_only + cross_only + both`.

### Canonical agent_id lumping
The librarian role in `*Librarian*` runner classes is a single
**persistent session** across wakes; all `librarian_<N>` raw agent_ids
collapse to canonical `librarian`. Other per-wake-fresh sub-agents
(BOAD `code_navigator`, HA `navigator/executor/editor`) retain
per-wake distinct canonical ids. **Cross between librarian wakes
is not a meaningful concept** — those reads fall under intra.

## Sub-agent invocation patterns

Per-episode mean # of distinct sub-agent canonical sessions
(orchestrator/planner excluded):

| family       | 27B       | 35B-A3B   |
|-------------|-----------|-----------|
| BOAD         | 2.0       | 1.2       |
| BOAD+lib     | 2.0       | 1.8       |
| HA           | 24.0      | 27.6      |
| HA+lib       | 17.6      | 23.6      |
| Librarian    | 2.0       | 2.0       |

→ BOAD delegates rarely (orch handles most lookups itself). HA delegates
aggressively (24+ fresh sub-agents per episode).

## (1A) BOAD orch ↔ orch intra redundant output

The orchestrator's own session re-reads files multiple times. Lib offloads
file lookup to the librarian; orch reads less and re-reads less.

| model | boad | boad+lib | Δ |
|---|---|---|---|
| 27B    | 486   | 282      | **−42%** |
| 35B-A3B | **1264** | **500**  | **−60%** |

## (1B) HA per-role-pair cross redundant output (canonical lumped)

### 27B HA / HA+lib
| pair                            | ha    | ha+lib |
|---------------------------------|-------|--------|
| **executor → executor**         | **1737** | **1428** |
| navigator → navigator           | 746   | (lumped to librarian → intra) |
| editor → navigator              | 140   | editor → lib: 179 |
| executor → navigator            | 107   | executor → lib: 59 |
| navigator → executor            | 37    | librarian → executor: 18 |
| editor → editor                 | 31    | 17 |
| navigator → editor              | 29    | librarian → editor: 24 |
| editor → executor               | 23    | 8 |
| **Total cross**                 | **2865** | **1733 (−39%)** |

### 35B HA / HA+lib
| pair                            | ha    | ha+lib |
|---------------------------------|-------|--------|
| **executor → executor**         | **1877** | **2527** ↑ |
| navigator → navigator           | 1093  | (lumped → intra) |
| executor → navigator            | 259   | executor → lib: 118 |
| editor → navigator              | 149   | editor → lib: 265 |
| navigator → executor            | 78    | librarian → executor: 33 |
| editor → editor                 | 39    | 29 |
| executor → editor               | 39    | 22 |
| editor → executor               | 29    | 5 |
| navigator → editor              | 23    | librarian → editor: 41 |
| **Total cross**                 | **3500** | **3040 (−13%)** |

### Key observation

- `navigator → navigator` cross (the dominant non-executor source) is
  fully **absorbed into intra** by canonical lumping — librarian's
  persistent session handles what was independently re-explored across
  navigator wakes.
- `executor → executor` cross is **untouched by lib** (35B HA actually
  rises by +35%). Reason: HyperAgent's executor inspects files via
  `bash` (`cat`, `head -N`, `sed -n A,Bp`), a tool channel the
  librarian sub-agent does not serve.

## (2) Per-role total token usage (input + output, avg per episode)

### BOAD orchestrator

| model | agent | in       | out  | total     | orch↔orch intra |
|---|---|----------|------|-----------|------------------|
| 27B    | boad      | 779k     | 6.7k | **786k**  | 486 |
| 27B    | boad+lib  | 659k     | 6.1k | **665k (−15%)** | 282 |
| 35B    | boad      | 1.49M    | 11k  | **1.50M** | 1264 |
| 35B    | boad+lib  | 983k     | 8.6k | **992k (−34%)** | 500 |

→ Lib reduces orch's total token consumption by **15-34%**, primarily by
shrinking input (less file content re-fed to orch).

### HA nav-role (navigator → librarian)

| model | agent | role      | in     | out    | total |
|---|---|-----------|--------|--------|-------|
| 27B    | ha       | navigator | 602k   | 10.3k  | **612k** |
| 27B    | ha+lib   | librarian | 517k   | **4.7k** | **521k (−15%)** |
| 35B    | ha       | navigator | 696k   | 12.8k  | **709k** |
| 35B    | ha+lib   | librarian | 846k   | **6.0k** | **852k (+20%)** ⚠️ |

Notable:
- **out_tokens halve** when navigator is replaced by librarian
  (`10.3k → 4.7k`, `12.8k → 6.0k`). Persistent session generates more
  concise outputs.
- On 35B, **librarian input grows +21%** above what navigator consumed
  (696k → 846k) — accumulated context across wakes inflates the
  prompt. Net total slightly increases (+20%) despite output halving.

## Key takeaways

1. **Lib's mechanism differs by architecture**:
   - **BOAD**: orchestrator-side intra reduction dominates. Orch
     delegates lookup to librarian, doesn't re-view files itself.
     (35B BOAD orch↔orch intra −60%).
   - **HA**: navigator-side cross absorption dominates. The many
     stateless `navigator_*` wakes collapse into one persistent
     librarian session — what was cross becomes intra and the
     per-wake re-exploration vanishes.

2. **Executor (HA) is lib's blind spot**: 60-75% of HA's residual cross
   redundancy is `executor → executor`, untouched by librarian. The
   executor's bash-based file inspection (`cat`, `head`, `sed -n`)
   bypasses the librarian channel.

3. **Persistent session output efficiency**: the same file-lookup role,
   when given persistent state, produces ~½ the output tokens
   (HA navigator → librarian out_tok −54%). The cost is input context
   accumulation (+20% on 35B HA).

4. **Sliding-window framing for HA**: HA's intra-redundancy includes
   substantial `sliding_fwd/bwd` patterns (~44-66% of intra-overlap
   pairs), reflecting chunk-by-chunk file scanning. Lib reduces this
   by holding prior chunks in persistent context.

5. **Caveats**:
   - cross comparison across architectures is **unfair**: BOAD delegates
     1-2× per episode (cross opportunities scarce), HA 24-28× (cross
     ubiquitous). Per-role-pair tables only meaningful within a family.
   - Strict-containment definition changes the picture: BOAD's
     redundancy is mostly `subset` (re-visits), HA's is mostly
     `sliding`. Under strict, BOAD shows larger lib effect (−30-39%);
     HA shows weaker (−11-17%).

## Source scripts

- `scripts/episode_meta.py` — episode-level metadata extraction (passes,
  tokens, failure modes, energy).
- `scripts/redundancy_full.py` — per-episode redundancy metrics
  (3-way decomp, conditional-prob inputs, turn distances, overlap
  patterns, per-role JSON).
- `scripts/episode_turns.py` — per-turn (per-LLM-call) row extraction.

## Output CSVs

- `/tmp/episodes.csv`  — 53 columns, 1 row per episode (5999 rows).
- `/tmp/turns.csv`     — 15 columns, 1 row per LLM call (720k rows).

Both keyed by `_path` (the source Evaluator JSON).
