# third_party — vendored read-only sources

Files in this directory are LITERAL COPIES from upstream open-source
projects, pinned to a specific commit. The intent is to gain access to
proven, validated implementations (stuck detection, the agent system
prompt, the persistent-shell bash tool) without forking the upstream
projects' full surface area.

## Rules

1. **Read-only.** Do not edit any vendored file. If you need to change
   behavior, write a thin adapter in `src/eval_pipeline/agents/bases/`
   that wraps the vendored function.
2. **Commit-pinned.** Each subdirectory has a `COMMIT_PIN.txt` recording
   the upstream commit SHA. Re-vendoring requires updating the pin AND
   re-running smoke tests against the new commit.
3. **No partial edits / line-deletions.** Either vendor a whole file or
   not at all. If only part of a file is needed, vendor the whole file
   and `from third_party.X import only_what_you_need`.
4. **CI guard** (TODO): a `pytest` check should diff vendored files
   against the upstream content at `COMMIT_PIN.txt` and fail on drift.

## Why vendor instead of `pip install`?

- Some upstream features sit inside private/internal modules that aren't
  exposed via the public package interface.
- `pip install` couples our env to upstream's full dependency tree;
  vendoring lets us cherry-pick the few files we actually use.
- We want to be confident our research results aren't an artifact of a
  silently-changed upstream version.

## Subdirectories

### `openhands_vendored/`
Source: https://github.com/All-Hands-AI/OpenHands @ `COMMIT_PIN.txt`
Files vendored (byte-identical to upstream):
- `stuck.py` — agent stuck-detection (V0 controller), ~488 lines.
  Used as the **algorithmic reference** for
  `src/eval_pipeline/agents/stuck_detection.py :: StuckDetector`, which
  ports the same five scenarios over a typed event hierarchy reachable
  from the openai-agents SDK. Not imported at runtime.
- `prompts/system_prompt.j2` — the CodeActAgent system prompt (MEMORY /
  EFFICIENCY / CODE_QUALITY / VERSION_CONTROL / PROBLEM_SOLVING_WORKFLOW /
  ... sections). **Loaded verbatim at runtime** by
  `src/eval_pipeline/agents/prompts/oh_system.py :: load_oh_system_prompt`.

(The OpenHands ACI file-editor truncation constants are reimplemented
inline in `src/eval_pipeline/agents/tools/_format.py` and `tools/oh/`,
so no `aci_editor/` copy is vendored here.)

### `openhands_legacy_vendored/`
Source: https://github.com/All-Hands-AI/OpenHands @ `COMMIT_PIN.txt`
- `bash.py` + `bash_constants.py` — the V0 persistent-shell bash session.
  Imported at runtime by `src/eval_pipeline/utils/docker.py` and
  `utils/_bash_helpers.py`.

## Usage pattern

For runtime-loaded files (system prompt, bash session), import from
`third_party.<dir>.*` directly. For algorithmic references (`stuck.py`),
keep the vendored file as a verifiable spec and implement the adapter
under `src/eval_pipeline/`.
