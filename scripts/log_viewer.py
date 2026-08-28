from __future__ import annotations

import argparse
import html
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _turn_compat import normalize_call  # noqa: E402


SED_READ   = re.compile(r"sed\s+(?:-n\s+)?['\"]?(\d+),(\d+)[p]['\"]?\s+(\S+)")
HEAD_RE    = re.compile(r"head\s+-n\s*(\d+)\s+(\S+)")
CAT_RE     = re.compile(r"(?:^|[&;|])\s*cat\s+(?![-<>])([^\s<>]+)")
SED_WRITE  = re.compile(r"sed\s+-i\b[^|;]*?\s(\S+)")
REDIRECT   = re.compile(r"(?:^|[^>&])>{1,2}\s*([^\s&][^\s]*)")
RM_RE      = re.compile(r"(?:^|[&;|])\s*rm\s+(?:-[rf]+\s+)?(\S+)")
MV_RE      = re.compile(r"(?:^|[&;|])\s*mv\s+\S+\s+(\S+)")
CP_RE      = re.compile(r"(?:^|[&;|])\s*cp\s+\S+\s+(\S+)")
TEE_RE     = re.compile(r"(?:^|[|&;])\s*tee\s+(\S+)")


def _parse_action(action: str) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    for piece in (action or "").split("; "):
        m = re.match(r"^(\w+)\((.*)\)$", piece, re.DOTALL)
        if not m:
            continue
        tool, raw = m.group(1), m.group(2)
        try:
            out.append((tool, json.loads(raw)))
        except Exception:
            out.append((tool, {"_raw": raw}))
    return out


def _extract_reads(action: str) -> list[tuple[str, int, int]]:
    out: list[tuple[str, int, int]] = []
    for tool, args in _parse_action(action):
        if tool == "bash":
            cmd = args.get("command") or ""
            for m in SED_READ.finditer(cmd):
                out.append((m.group(3), int(m.group(1)), int(m.group(2))))
            for m in HEAD_RE.finditer(cmd):
                out.append((m.group(2), 1, int(m.group(1))))
            for m in CAT_RE.finditer(cmd):
                out.append((m.group(1), 1, -1))
        elif tool == "str_replace_editor":
            if args.get("command") == "view":
                p = args.get("path") or ""
                vr = args.get("view_range") or []
                if isinstance(vr, str):
                    try: vr = json.loads(vr)
                    except Exception: vr = []
                if isinstance(vr, list) and len(vr) == 2 and vr[0] > 0:
                    out.append((p, int(vr[0]), int(vr[1])))
                else:
                    out.append((p, 1, -1))
    return out


_BASH_TRUNC_MARK = "[Previous command outputs are truncated"
_EDITOR_TRUNC_MARK = "<response clipped>"

_REJECTED_MARK = "REJECTED:"
_PYDANTIC_MARK = "Invalid JSON input for tool"


def _is_truncated_obs(obs: str) -> bool:
    return _BASH_TRUNC_MARK in obs[:200] or _EDITOR_TRUNC_MARK in obs


_NO_OUTPUT_RE = re.compile(r"^\s*\(command exited with code -?\d+\)\s*$")


def _is_failed_obs(obs: str) -> bool:
    head = obs[:300]
    if _REJECTED_MARK in head or _PYDANTIC_MARK in head:
        return True
    return bool(_NO_OUTPUT_RE.match(obs))


def _parse_sweagent_action_str(action: str) -> list[dict]:
    s = action.strip()
    if not s:
        return []
    KNOWN = ("str_replace_editor", "submit_subagent", "submit",
             "issue_analyzer", "code_navigator", "filemap", "search")
    head = s.split(maxsplit=1)[0]
    if head == "str_replace_editor":
        rest = s.split(maxsplit=2)[1:]
        cmd = rest[0] if rest else ""
        rest2 = s.split(maxsplit=2)[2] if len(s.split(maxsplit=2)) > 2 else ""
        m = re.match(r"(\S+)(?:\s+\[(\d+),\s*(-?\d+)\])?", rest2)
        path = m.group(1) if m else ""
        args: dict = {"command": cmd, "path": path}
        if m and m.group(2) is not None:
            args["view_range"] = [int(m.group(2)), int(m.group(3))]
        return [{"name": "str_replace_editor",
                 "arguments": json.dumps(args)}]
    if head in KNOWN:
        body = s.split(maxsplit=1)[1] if " " in s else ""
        return [{"name": head, "arguments": json.dumps({"context": body})}]
    return [{"name": "bash", "arguments": json.dumps({"command": s})}]


def _extract_reads_with_obs(
    raw_action, raw_obs,
) -> list[dict]:
    action_items = raw_action if isinstance(raw_action, list) else []
    if not action_items and isinstance(raw_action, str) and raw_action.strip():
        action_items = _parse_sweagent_action_str(raw_action)
    obs_items = raw_obs if isinstance(raw_obs, list) else []
    if not obs_items and isinstance(raw_obs, str) and raw_obs:
        obs_items = [{"tool_name": "?", "output": raw_obs}]

    out: list[dict] = []
    obs_idx = 0
    for a in action_items:
        if not isinstance(a, dict): continue
        name = a.get("name", "")
        try: args = json.loads(a.get("arguments") or "{}")
        except Exception: args = {}
        obs_text = ""
        if obs_idx < len(obs_items):
            o = obs_items[obs_idx]
            if isinstance(o, dict):
                obs_text = o.get("output", "") or ""
            obs_idx += 1

        reads_in_action: list[str] = []
        if name == "bash":
            cmd = args.get("command") or ""
            for m in SED_READ.finditer(cmd):
                reads_in_action.append(m.group(3))
            for m in HEAD_RE.finditer(cmd):
                reads_in_action.append(m.group(2))
            for m in CAT_RE.finditer(cmd):
                reads_in_action.append(m.group(1))
        elif name == "str_replace_editor":
            if args.get("command") == "view":
                p = args.get("path") or ""
                if p:
                    reads_in_action.append(p)

        if not reads_in_action: continue

        failed = _is_failed_obs(obs_text)
        truncated = _is_truncated_obs(obs_text)
        is_dir_listing = (
            "files and directories up to 2 levels deep" in obs_text[:200]
        )

        for fpath in reads_in_action:
            if is_dir_listing:
                continue
            out.append({
                "file": fpath,
                "truncated": truncated,
                "failed": failed,
            })
    return out


def _extract_writes(action: str) -> list[str]:
    out: list[str] = []
    for tool, args in _parse_action(action):
        if tool == "bash":
            cmd = args.get("command") or ""
            for pat in (SED_WRITE, REDIRECT, RM_RE, MV_RE, CP_RE, TEE_RE):
                for m in pat.finditer(cmd):
                    f = m.group(1)
                    if not f.startswith("&"):
                        out.append(f)
        elif tool == "str_replace_editor":
            if args.get("command") in {"str_replace", "create", "insert"}:
                p = args.get("path")
                if p:
                    out.append(p)
    return out


_EDITOR_ERROR_RE = re.compile(r"^\s*(Error|ERROR):", re.MULTILINE)


def _did_write_succeed(tool: str, obs_text: str) -> bool:
    if not obs_text:
        return True
    if _is_failed_obs(obs_text):
        return False
    if tool == "str_replace_editor" and _EDITOR_ERROR_RE.search(obs_text[:300]):
        return False
    return True


def _extract_writes_with_obs(action, observation) -> list[str]:
    if isinstance(observation, list):
        obs_items = observation
    elif isinstance(observation, str):
        obs_items = [{"output": observation}]
    else:
        obs_items = []

    if isinstance(action, list):
        action_items = action
    elif isinstance(action, str):
        action_items = _parse_sweagent_action_str(action)
    else:
        action_items = []

    out: list[str] = []
    obs_idx = 0
    for a in action_items:
        if not isinstance(a, dict):
            continue
        name = a.get("name", "")
        try:
            args = json.loads(a.get("arguments") or "{}")
        except Exception:
            args = {}
        obs_text = ""
        if obs_idx < len(obs_items):
            o = obs_items[obs_idx]
            if isinstance(o, dict):
                obs_text = o.get("output", "") or ""
            obs_idx += 1

        wrote_paths: list[str] = []
        if name == "bash":
            cmd = args.get("command") or ""
            for pat in (SED_WRITE, REDIRECT, RM_RE, MV_RE, CP_RE, TEE_RE):
                for m in pat.finditer(cmd):
                    f = m.group(1)
                    if not f.startswith("&"):
                        wrote_paths.append(f)
        elif name == "str_replace_editor":
            if args.get("command") in {"str_replace", "create", "insert"}:
                p = args.get("path")
                if p:
                    wrote_paths.append(p)

        if wrote_paths and _did_write_succeed(name, obs_text):
            out.extend(wrote_paths)
    return out


def _action_summary(action: str) -> str:
    parsed = _parse_action(action)
    if not parsed:
        return action[:60] if action else "(none)"
    parts = []
    for tool, args in parsed:
        if tool == "bash":
            cmd = (args.get("command") or "").split("\n", 1)[0]
            parts.append(f"bash: {cmd[:90]}")
        elif tool == "str_replace_editor":
            parts.append(
                f"editor:{args.get('command','?')} "
                f"{(args.get('path') or '').split('/')[-1]} "
                f"{args.get('view_range', '')}"
            )
        elif tool == "submit_result":
            parts.append(f"SUBMIT: {(args.get('result') or '')[:80]}…")
        elif tool == "submit":
            parts.append(f"SUBMIT (orchestrator): {(args.get('result') or '')[:80]}…")
        elif tool == "ask_file_agent":
            parts.append(
                f"ASK→{(args.get('file_path') or '').split('/')[-1]}: "
                f"{(args.get('query') or '')[:80]}…"
            )
        else:
            parts.append(f"{tool}({str(args)[:60]})")
    return "; ".join(parts)


def build_turns(calls: list[dict]) -> list[dict]:
    turns: list[dict] = []
    for i, c in enumerate(calls):
        raw_action = c.get("action")
        if isinstance(raw_action, list):
            parts = []
            for tc in raw_action:
                if isinstance(tc, dict):
                    name = tc.get("name", "?")
                    args = tc.get("arguments", "")
                    parts.append(f"{name}({args})")
            action = "; ".join(parts)
        else:
            action = raw_action or ""

        raw_obs = c.get("observation")
        if isinstance(raw_obs, list):
            obs_str = "\n---\n".join(
                f"[{o.get('tool_name','?')}] {o.get('output','')}"
                if isinstance(o, dict) else str(o)
                for o in raw_obs
            )
        else:
            obs_str = raw_obs or ""

        text_str = c.get("text") or c.get("response") or ""

        reads = list(_extract_reads_with_obs(c.get("action"), c.get("observation")))
        successful_writes = _extract_writes_with_obs(
            c.get("action"), c.get("observation"),
        )
        turns.append({
            "idx": i,
            "agent_id": c.get("agent_id") or "?",
            "agent_role": c.get("agent_role") or "?",
            "duration": c.get("duration_seconds"),
            "input_tokens": c.get("input_tokens"),
            "output_tokens": c.get("output_tokens"),
            "cached_tokens": c.get("cached_tokens"),
            "cache_q_delta": c.get("cache_q_delta"),
            "cache_h_delta": c.get("cache_h_delta"),
            "cache_hit_ratio": c.get("cache_hit_ratio"),
            "joules": c.get("joules"),
            "summary": _action_summary(action),
            "action": action,
            "text": text_str,
            "reads": reads,
            "writes": successful_writes,
            "observation": obs_str,
            "error": c.get("error"),
            "model": c.get("model"),
            "attempt": c.get("attempt"),
            "raw_prompt": c.get("raw_prompt"),
            "raw_completion": c.get("raw_completion"),
            "reasoning": c.get("reasoning"),
            "input_meta": (
                json.dumps(c.get("input"), default=str, ensure_ascii=False)
                if isinstance(c.get("input"), list)
                else (c.get("input") or "")
            ),
        })

    prev_in_tok: dict[str, int] = {}
    prev_input_list: dict[str, list] = {}
    prev_input_str: dict[str, str] = {}
    for i_idx, t in enumerate(turns):
        aid = t["agent_id"]

        prev = prev_in_tok.get(aid)
        cur = t["input_tokens"] or 0
        t["input_tokens_delta"] = (cur - prev) if prev is not None else cur
        prev_in_tok[aid] = cur

        raw_inp = calls[i_idx].get("input")
        if isinstance(raw_inp, list):
            prev_n = len(prev_input_list.get(aid, []))
            new_items = raw_inp[prev_n:]
            new_items = [
                it for it in new_items
                if not (
                    isinstance(it, dict)
                    and (it.get("type") == "function_call"
                         or (it.get("type") == "message"
                             and it.get("role") == "assistant"))
                )
            ]
            prev_input_list[aid] = list(raw_inp)
            t["input_meta_new"] = json.dumps(new_items, default=str, ensure_ascii=False)
        else:
            cur_inp = raw_inp or ""
            prev_inp = prev_input_str.get(aid, "")
            n = min(len(cur_inp), len(prev_inp))
            j = 0
            while j < n and cur_inp[j] == prev_inp[j]:
                j += 1
            t["input_meta_new"] = cur_inp[j:]
            prev_input_str[aid] = cur_inp

    return turns


def _render_input(raw: str) -> str:
    if not raw:
        return "(none)"
    try:
        data = json.loads(raw)
    except Exception:
        return html.escape(raw)
    if isinstance(data, list):
        out = []
        for m in data:
            if isinstance(m, dict):
                role = m.get("role", "?")
                if m.get("type") == "function_call":
                    name = m.get("name", "?")
                    args = m.get("arguments", "")
                    out.append(f"[tool_call: {name}]\n{args}")
                elif m.get("type") == "function_call_output":
                    out.append(f"[tool_output]\n{m.get('output', '')}")
                else:
                    content = m.get("content", "")
                    if isinstance(content, list):
                        content = "\n".join(
                            (b.get("text") if isinstance(b, dict) else str(b))
                            for b in content
                        )
                    out.append(f"[{role}]\n{content}")
            else:
                out.append(str(m))
        return html.escape("\n\n".join(out))
    elif isinstance(data, dict):
        return html.escape(json.dumps(data, indent=2, ensure_ascii=False))
    return html.escape(str(data))


CSS = """
body { font-family: -apple-system, system-ui, sans-serif; font-size: 13px;
       margin: 0; padding: 16px; background: #fafafa; }
.toolbar { position: sticky; top: 0; background: #fafafa; padding: 8px 0;
           border-bottom: 1px solid #ddd; margin-bottom: 8px; z-index: 10; }
.toolbar select { font-size: 13px; padding: 4px 8px; }
.summary-bar { background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
               padding: 8px 12px; margin: 8px 0; }
.bar-row { display: flex; gap: 24px; }
.bar-row .stat b { color: #111; }
.turn { background: #fff; border: 1px solid #ddd; border-radius: 4px;
        margin: 8px 0; }
.turn .head { padding: 6px 10px; background: #f6f6f6;
              border-bottom: 1px solid #e8e8e8; font-size: 12px; }
.turn .head b { color: #222; }
.turn .head .meta { color: #888; margin-left: 10px; }
.box { margin: 6px 10px; }
.box-label { font-size: 10px; color: #888; text-transform: uppercase;
             letter-spacing: 0.05em; margin-bottom: 2px; }
.box-body { white-space: pre-wrap; font-family: ui-monospace, Menlo, monospace;
            font-size: 11px; padding: 6px 8px; border-radius: 3px;
            max-height: 500px; overflow: auto; }
.box-input  .box-body { background: #f0f4fa; color: #234; }
.box-output .box-body { background: #fffbe8; color: #432; }
.box-obs    .box-body { background: #f3f3f3; color: #333; }
.box-raw    .box-body { background: #1e1e1e; color: #d4d4d4; }
.box-raw    summary   { cursor: pointer; user-select: none; padding: 4px 8px;
                        background: #2d2d2d; color: #aaa; font-size: 10px;
                        text-transform: uppercase; letter-spacing: 0.05em;
                        border-radius: 3px; }
.box-raw    summary:hover { color: #fff; }
.box-raw    pre { margin: 0; padding: 6px 8px; font-family: ui-monospace, Menlo, monospace;
                  font-size: 11px; max-height: 600px; overflow: auto;
                  background: #1e1e1e; color: #d4d4d4; white-space: pre-wrap;
                  word-break: break-word; border-radius: 0 0 3px 3px; }
.role-orchestrator { border-left: 4px solid #1f77b4; }
.role-file_agent   { border-left: 4px solid #2ca02c; }
.role-planner      { border-left: 4px solid #ff7f0e; }
.role-other        { border-left: 4px solid #aaa; }

/* Evaluation panel */
.ev-panel { background: #fff; border: 1px solid #e0e0e0; border-radius: 4px;
            padding: 10px 12px; margin: 8px 0; }
.ev-pills { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
.ev-pill { display: inline-block; padding: 2px 8px; border-radius: 12px;
           font-size: 11px; background: #f0f0f0; color: #444; border: 1px solid #d8d8d8; }
.ev-pill b { color: #222; }
.ev-pill.ev-empty { opacity: 0.6; }
.ev-f2p { background: #e8f5e9; border-color: #b9e0bd; }
.ev-p2p { background: #e3f2fd; border-color: #b8dcf2; }
.ev-f2f { background: #fff8e1; border-color: #f5e2a8; }
.ev-p2f { background: #ffebee; border-color: #f5b7b7; color: #b00020; }
.ev-ok  { color: #2e7d32; }
.ev-bad { color: #c62828; }
.ev-bucket { margin: 4px 0; border: 1px solid #eee; border-radius: 3px; }
.ev-bucket > summary { cursor: pointer; user-select: none; padding: 5px 8px;
                       background: #fafafa; font-size: 12px; }
.ev-bucket > summary:hover { background: #f0f0f0; }
.ev-counts { color: #666; margin-left: 6px; font-size: 11px; }
.ev-tip    { color: #888; margin-left: 8px; font-size: 11px; font-style: italic; }
.ev-list { margin: 0; padding: 6px 12px; max-height: 280px; overflow: auto;
           list-style: none; font-family: ui-monospace, Menlo, monospace; font-size: 11px; }
.ev-list li { padding: 1px 0; }
.ev-summary { margin: 0; padding: 6px 8px; background: #fff5f5; color: #722;
              font-family: ui-monospace, Menlo, monospace; font-size: 11px;
              max-height: 240px; overflow: auto; white-space: pre-wrap; }
.ev-raw { margin: 0; padding: 6px 8px; background: #1e1e1e; color: #d4d4d4;
          font-family: ui-monospace, Menlo, monospace; font-size: 11px;
          max-height: 600px; overflow: auto; white-space: pre-wrap;
          word-break: break-word; }
.ev-err { background: #ffebee; color: #b00020; padding: 6px 10px; border-radius: 3px;
          margin: 6px 0; font-size: 12px; font-family: ui-monospace, Menlo, monospace; }
"""

JS = """
function setFilter(v) {
  document.querySelectorAll('.turn').forEach(el => {
    if (v === 'all' || el.dataset.aid === v) el.style.display = '';
    else el.style.display = 'none';
  });
}
"""


def _render_eval_panel(meta: dict) -> str:
    md = meta.get("eval_metadata") or {}
    ts = md.get("tests_status") or {}
    raw_eval = meta.get("eval_raw_output") or ""
    err = meta.get("agent_error") or ""

    bucket_order = [
        ("FAIL_TO_PASS", "f2p", "want pass: was failing → should now pass"),
        ("PASS_TO_PASS", "p2p", "regression check: was passing → should still pass"),
        ("FAIL_TO_FAIL", "f2f", "still-failing tests: not graded as regressions"),
        ("PASS_TO_FAIL", "p2f", "regressions: were passing → now fail (BAD)"),
    ]

    if not ts and not raw_eval and not err:
        return ""

    pills = []
    sections = []
    for key, klass, tip in bucket_order:
        b = ts.get(key) or {}
        ok = b.get("success") or []
        fail = b.get("failure") or []
        if not ok and not fail:
            pills.append(
                f"<span class='ev-pill ev-{klass} ev-empty' title='{tip}'>"
                f"{key}: 0/0</span>"
            )
            continue
        total = len(ok) + len(fail)
        pills.append(
            f"<span class='ev-pill ev-{klass}' title='{tip}'>"
            f"{key}: <b>{len(ok)}</b>/{total}"
            f"{' (' + str(len(fail)) + ' failed)' if fail else ''}"
            f"</span>"
        )
        ok_html = "".join(
            f"<li class='ev-ok'>✓ {html.escape(t)}</li>"
            for t in ok
        )
        fail_html = "".join(
            f"<li class='ev-bad'>✗ {html.escape(t)}</li>"
            for t in fail
        )
        sections.append(
            f"<details class='ev-bucket'>"
            f"<summary><b>{key}</b> "
            f"<span class='ev-counts'>({len(ok)} pass / {len(fail)} fail)</span>"
            f"<span class='ev-tip'>{tip}</span></summary>"
            f"<ul class='ev-list'>{fail_html}{ok_html}</ul></details>"
        )

    pir = (md.get("per_instance_report") or {})
    pir_inst = next(iter(pir.values())) if pir else {}
    patch_bits = []
    for k in ("patch_exists", "patch_successfully_applied", "resolved"):
        if k in pir_inst:
            v = pir_inst[k]
            css = "ev-ok" if v else "ev-bad"
            patch_bits.append(
                f"<span class='ev-pill {css}' title='{k}'>"
                f"{k}={'Y' if v else 'N'}</span>"
            )
    if patch_bits:
        pills = patch_bits + pills

    test_log = ""
    if "=== test runner stdout ===" in raw_eval:
        try:
            test_log = raw_eval.split(
                "=== test runner stdout ===", 1
            )[1].split("=== summary report ===", 1)[0].strip()
        except Exception:
            test_log = ""
    if not test_log and raw_eval:
        test_log = raw_eval[:50_000]
    test_log_html = ""
    if test_log:
        important_lines: list[str] = []
        for line in test_log.splitlines():
            ls = line.strip()
            if (ls.startswith("FAILED ") or ls.startswith("ERROR ")
                    or ls.startswith("FAIL: ") or ls.startswith("ERROR: ")
                    or " ... ok" in ls or " ... FAIL" in ls
                    or " ... ERROR" in ls):
                important_lines.append(line)
                if len(important_lines) > 200:
                    break
        summary_pre = (
            f"<pre class='ev-summary'>{html.escape(chr(10).join(important_lines))}</pre>"
            if important_lines else ""
        )
        test_log_html = (
            f"<details class='ev-bucket'>"
            f"<summary><b>test_output.txt</b> "
            f"<span class='ev-counts'>({len(test_log):,} chars)</span>"
            f"</summary>"
            f"{summary_pre}"
            f"<pre class='ev-raw'>{html.escape(test_log[:200_000])}"
            f"{('...[clipped]' if len(test_log) > 200_000 else '')}</pre>"
            f"</details>"
        )

    err_html = (
        f"<div class='ev-err' title='agent_error'>"
        f"agent_error: {html.escape(err[:300])}</div>"
        if err else ""
    )

    eval_err_label = _detect_eval_error(raw_eval)
    eval_err_html = ""
    if eval_err_label:
        snippet = raw_eval[:1200]
        eval_err_html = (
            f"<div style='background:#fde;border:1px solid #b35;color:#b35;"
            f"padding:8px 12px;margin:8px 0;border-radius:4px;'>"
            f"<b>⚠ EVAL-SIDE ERROR ({html.escape(eval_err_label)})</b> — "
            f"agent's patch was NOT graded; this is an evaluator infra "
            f"failure (not an agent fail). Re-run with "
            f"<code>scripts/reevaluate.py --path &lt;episode.json&gt;</code>."
            f"<details style='margin-top:6px'><summary>raw output (first 1.2 KB)</summary>"
            f"<pre style='white-space:pre-wrap;font-size:12px;margin:4px 0 0 0'>"
            f"{html.escape(snippet)}</pre></details></div>"
        )

    return f"""
<div class='ev-panel'>
  {eval_err_html}
  <div class='ev-pills'>{''.join(pills)}</div>
  {err_html}
  {''.join(sections)}
  {test_log_html}
</div>
"""


def render_html(meta: dict, turns: list[dict]) -> str:
    agent_ids = sorted({t["agent_id"] for t in turns})
    options = "".join(
        f'<option value="{html.escape(a)}">{html.escape(a)}</option>'
        for a in agent_ids
    )

    _ene = meta.get("energy") or {}
    _total_J = (_ene.get("total_joules") or 0) / 1000
    _details = _ene.get("details") or {}
    _raw_J = (_details.get("total_joules_raw") or 0) / 1000
    _raw_html = (
        f"<div class='stat' style='background:#f0f4fa;color:#234;'>"
        f"kJ raw (NVML diff): <b>{_raw_J:.1f}</b></div>"
    ) if _raw_J and abs(_raw_J - _total_J) > 0.01 * max(_total_J, 0.1) else ""
    bar = (
        "<div class='summary-bar'><div class='bar-row'>"
        f"<div class='stat'><b>{html.escape(meta.get('task_id',''))}</b></div>"
        f"<div class='stat'>calls: <b>{len(turns)}</b></div>"
        f"<div class='stat'>pass: <b>{meta.get('passed')}</b></div>"
        f"<div class='stat'>kJ: <b>{_total_J:.1f}</b></div>"
        f"{_raw_html}"
        "</div></div>"
    )

    eval_html = _render_eval_panel(meta)

    ep_path = meta.get("_ep_path", "")
    rows = []
    for t in turns:
        role_cls = (
            f"role-{t['agent_role']}"
            if t["agent_role"] in {"orchestrator", "file_agent", "planner"}
            else "role-other"
        )
        in_tok = t.get("input_tokens") or 0
        delta = t.get("input_tokens_delta") or 0
        out_tok = t.get("output_tokens") or 0
        chr_ = t.get("cache_hit_ratio")
        if chr_ is not None:
            hit_d = int(t.get("cache_h_delta") or 0)
            q_d = int(t.get("cache_q_delta") or 0)
            cache_str = f"cache={chr_*100:.0f}% ({hit_d}/{q_d})"
        else:
            cache_str = "cache=—"
        rows.append(f"""
        <div class='turn {role_cls}' data-aid='{html.escape(t['agent_id'])}' data-idx='{t['idx']}'>
          <div class='head' onclick='toggleTurn(this)'>
            <span class='caret'>▶</span>
            #{t['idx']} · <b>{html.escape(t['agent_id'])}</b>
            <span class='meta'>in={in_tok} (+{delta}) · {cache_str} · out={out_tok} · dur={(t.get('duration') or 0):.1f}s · J={t.get('joules') or 0:.0f}</span>
          </div>
          <div class='turn-body' style='display:none'></div>
        </div>
        """)

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>{html.escape(meta.get('task_id',''))} — log</title>
<style>{CSS}
.turn .head {{ cursor: pointer; user-select: none; }}
.turn .head:hover {{ background: #f0f0f0; }}
.turn .caret {{ display: inline-block; transition: transform 150ms; margin-right: 4px; color:#888; }}
.turn.open .caret {{ transform: rotate(90deg); }}
</style></head>
<body data-ep-path='{html.escape(ep_path)}'>
<div class='toolbar'>
  <label>filter agent_id:
    <select onchange='setFilter(this.value)'>
      <option value='all'>(all)</option>
      {options}
    </select>
  </label>
  <button onclick='expandAll()' style='margin-left:14px'>expand all</button>
  <button onclick='collapseAll()'>collapse all</button>
</div>
{bar}
{eval_html}
{''.join(rows)}
<script>{JS}
async function toggleTurn(headDiv) {{
  const turn = headDiv.parentElement;
  const body = turn.querySelector('.turn-body');
  if (turn.classList.contains('open')) {{
    body.style.display = 'none';
    turn.classList.remove('open');
    return;
  }}
  if (!body.dataset.loaded) {{
    body.innerHTML = "<div style='padding:10px; color:#888'>loading…</div>";
    const path = document.body.dataset.epPath;
    const idx = turn.dataset.idx;
    try {{
      const r = await fetch(`/turn?path=${{encodeURIComponent(path)}}&idx=${{idx}}`);
      body.innerHTML = await r.text();
      body.dataset.loaded = '1';
    }} catch (e) {{
      body.innerHTML = "<div style='padding:10px; color:#a00'>error: " + e + "</div>";
    }}
  }}
  body.style.display = '';
  turn.classList.add('open');
}}
async function expandAll() {{
  for (const turn of document.querySelectorAll('.turn')) {{
    if (turn.classList.contains('open')) continue;
    if (turn.style.display === 'none') continue;   // skip filtered-out
    await toggleTurn(turn.querySelector('.head'));
  }}
}}
function collapseAll() {{
  document.querySelectorAll('.turn.open').forEach(t => {{
    t.querySelector('.turn-body').style.display = 'none';
    t.classList.remove('open');
  }});
}}
</script>
</body></html>
"""


def _render_turn_body(t: dict) -> str:
    output_parts = []
    if t.get("text"): output_parts.append(t["text"])
    if t.get("action"): output_parts.append(t["action"])
    output_text = "\n\n".join(output_parts) or "(no output)"

    raw_blocks = ""
    rsn = t.get("reasoning")
    if isinstance(rsn, str) and rsn:
        raw_blocks += (
            "<div class='box box-raw'>"
            "<details><summary>reasoning — chain-of-thought "
            f"({len(rsn)} chars)</summary>"
            f"<pre>{html.escape(rsn)}</pre>"
            "</details></div>"
        )
    rp = t.get("raw_prompt"); rc = t.get("raw_completion")
    if isinstance(rp, str) and rp:
        raw_blocks += (
            "<div class='box box-raw'>"
            "<details><summary>raw_prompt — chat-template rendered "
            f"({len(rp)} chars)</summary>"
            f"<pre>{html.escape(rp)}</pre>"
            "</details></div>"
        )
    if isinstance(rc, str) and rc:
        raw_blocks += (
            "<div class='box box-raw'>"
            "<details><summary>raw_completion — pre tool-parser "
            f"({len(rc)} chars)</summary>"
            f"<pre>{html.escape(rc)}</pre>"
            "</details></div>"
        )
    return f"""
    <div class='box box-input'>
      <div class='box-label'>input (new since prior {html.escape(t['agent_id'])} turn)</div>
      <div class='box-body'>{_render_input(t.get('input_meta_new') or '')}</div>
    </div>
    <div class='box box-output'>
      <div class='box-label'>output</div>
      <div class='box-body'>{html.escape(output_text)}</div>
    </div>
    <div class='box box-obs'>
      <div class='box-label'>observation</div>
      <div class='box-body'>{html.escape(t.get('observation') or '(none)')}</div>
    </div>
    {raw_blocks}
    """


def _list_episodes(matrix_dir: Path) -> list[dict]:
    entries = []
    for task_dir in sorted(matrix_dir.iterdir()):
        if not task_dir.is_dir() or task_dir.name.startswith("_"):
            continue
        for agent_dir in sorted(task_dir.iterdir()):
            if not agent_dir.is_dir():
                continue
            final = sorted(
                p for p in agent_dir.glob("*Evaluator*.json")
                if not p.name.endswith(".partial.json")
            )
            partial = sorted(agent_dir.glob("*Evaluator*.partial.json"))
            live = agent_dir / "live_calls.jsonl"
            if final:
                entries.append({
                    "task": task_dir.name,
                    "agent_seed": agent_dir.name,
                    "path": str(final[-1]),
                    "in_progress": False,
                })
            elif partial:
                entries.append({
                    "task": task_dir.name,
                    "agent_seed": agent_dir.name,
                    "path": str(partial[-1]),
                    "in_progress": True,
                    "phase": "eval-pending",
                })
            elif live.exists():
                entries.append({
                    "task": task_dir.name,
                    "agent_seed": agent_dir.name,
                    "path": str(live),
                    "in_progress": True,
                    "phase": "agent-live",
                })
    return entries


def _merge_raw_http(calls: list[dict], http_path: Path) -> None:
    if not http_path.exists():
        return
    http_recs: list[dict] = []
    for line in http_path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            http_recs.append(json.loads(line))
        except Exception:
            pass
    for i, c in enumerate(calls):
        if i >= len(http_recs):
            break
        h = http_recs[i]
        if "raw_prompt" in h:
            c["raw_prompt"] = h.get("raw_prompt")
        if "raw_completion" in h:
            c["raw_completion"] = h.get("raw_completion")
        if "reasoning" in h:
            c["reasoning"] = h.get("reasoning")


def _load_episode(json_path: Path) -> dict:
    text = json_path.read_text()
    if json_path.name.endswith(".jsonl"):
        calls = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                calls.append(json.loads(line))
            except Exception:
                pass
        calls = [normalize_call(c) for c in calls]
        _merge_raw_http(calls, json_path.parent / "live_http.jsonl")
        return {
            "task_id": json_path.parent.parent.name,
            "passed": None,
            "energy": {
                "total_joules": sum(c.get("joules") or 0 for c in calls),
                "total_joules_clean": _derive_clean_energy(calls)[0],
                "n_outlier_calls": _derive_clean_energy(calls)[1],
            },
            "llm_calls": calls,
            "_live": True,
        }
    data = json.loads(text)
    data["llm_calls"] = [
        normalize_call(c) for c in (data.get("llm_calls") or [])
    ]
    _merge_raw_http(data["llm_calls"], json_path.parent / "live_http.jsonl")
    return data


def _calls_metrics(calls: list[dict], task_id: str | None = None) -> dict:
    if not calls:
        return {"max_agent_ctx": None,
                "n_agents": 0, "n_reads_truncated": 0}
    n_trunc = 0
    try:
        turns = build_turns(calls)
        for t in turns:
            for rd in t.get("reads") or []:
                if rd.get("failed"): continue
                if rd.get("truncated"): n_trunc += 1
    except Exception:
        pass
    max_agent_ctx = 0
    agent_ids: set[str] = set()
    for c in calls:
        n = c.get("input_tokens") or 0
        if n > max_agent_ctx: max_agent_ctx = n
        aid = c.get("agent_id")
        if aid: agent_ids.add(aid)
    return {
        "max_agent_ctx": max_agent_ctx or None,
        "n_agents": len(agent_ids),
        "n_reads_truncated": n_trunc,
    }


_STATS_CACHE: dict[str, tuple[float, dict]] = {}

_TURNS_CACHE: dict[str, tuple[float, list]] = {}


def _get_episode_turns(ev_path: Path) -> list:
    key = str(ev_path)
    try:
        mtime = ev_path.stat().st_mtime
    except FileNotFoundError:
        return []
    cached = _TURNS_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    meta = _load_episode(ev_path)
    turns = build_turns(meta.get("llm_calls") or [])
    _TURNS_CACHE[key] = (mtime, turns)
    return turns


def _episode_stats(ev_path: Path) -> dict:
    """Per-episode metrics derived from Evaluator JSON (or live jsonl).
    Includes reads/line + max context windows (orchestrator / sub-agent)
    via ``_calls_metrics``. Memoized by (path, mtime)."""
    key = str(ev_path)
    try:
        mtime = ev_path.stat().st_mtime
    except FileNotFoundError:
        mtime = 0.0
    cached = _STATS_CACHE.get(key)
    if cached and cached[0] == mtime:
        return cached[1]
    stats = _episode_stats_uncached(ev_path)
    _STATS_CACHE[key] = (mtime, stats)
    return stats


def _derive_clean_energy(calls: list[dict]) -> tuple[float, int]:
    total = 0.0
    for c in calls:
        j = c.get("joules") or 0.0
        total += float(j)
    return total, 0


def _detect_eval_error(eval_raw_output: str) -> str | None:
    """Return a short eval-side-error label if the harness/evaluation
    crashed (vs. agent legitimately failing the test). Used by both
    the viewer (status pill) and ``scripts/reevaluate.py`` (which
    re-runs eval for episodes flagged this way).

    Detection markers:
      - "Harness failed (rc=…)": SWEBenchLiveEvaluator's wrapper around
        the vendored SWE-bench-Live harness subprocess.
      - "Harness timed out": SWEBenchLiveEvaluator's 6000s wall clamp.
      - "ConnectionError" / "Name or service not known": DNS / network
        layer crash inside the swebench harness's GitHub raw fetch
        (see ``swebench.harness.test_spec.python.get_environment_yml``).
      - "No patch generated": empty submission, NOT an eval crash —
        intentionally NOT flagged here so legit empty-patch fails
        stay as plain ``passed=False``.
    """
    if not eval_raw_output:
        return None
    if eval_raw_output.startswith("Harness failed"):
        if "ConnectionError" in eval_raw_output or "Name or service not known" in eval_raw_output:
            return "harness_dns_error"
        if "Max retries exceeded" in eval_raw_output:
            return "harness_network_error"
        return "harness_subprocess_error"
    if eval_raw_output.startswith("Harness timed out"):
        return "harness_timeout"
    if eval_raw_output.startswith("Report not found"):
        return "harness_no_report"
    if "ConnectionError" in eval_raw_output and "raw.githubusercontent.com" in eval_raw_output:
        return "harness_dns_error"
    return None


def _episode_stats_uncached(ev_path: Path) -> dict:
    try:
        if ev_path.name.endswith(".jsonl"):
            calls = [json.loads(l) for l in ev_path.read_text().splitlines() if l.strip()]
            task_id = ev_path.parent.parent.name if ev_path.parent.parent else None
            m = _calls_metrics(calls, task_id=task_id)
            clean_J, n_out = _derive_clean_energy(calls)
            return {
                "passed": None,
                "score": None,
                "eval_error": None,
                "turns": len(calls),
                "duration": sum(c.get("duration_seconds") or 0 for c in calls),
                "kJ_clean": clean_J / 1000,
                "n_outliers": n_out,
                "input_tokens": sum(c.get("input_tokens") or 0 for c in calls),
                "output_tokens": sum(c.get("output_tokens") or 0 for c in calls),
                "cached_tokens": sum(c.get("cache_h_delta") or 0 for c in calls),
                "patch_len": 0,
                "in_progress": True,
                "phase": "agent-live",
                **m,
            }
        is_partial = ev_path.name.endswith(".partial.json")
        d = json.loads(ev_path.read_text())
        calls = d.get("llm_calls") or []
        m = _calls_metrics(calls, task_id=d.get("task_id"))
        energy = d.get("energy") or {}
        clean_J = energy.get("total_joules_clean")
        n_out = energy.get("n_outlier_calls")
        if clean_J is None or n_out is None:
            clean_J_calc, n_out_calc = _derive_clean_energy(calls)
            if clean_J is None: clean_J = clean_J_calc
            if n_out is None: n_out = n_out_calc
        tu = d.get("token_usage") or {}
        eval_error = _detect_eval_error(d.get("eval_raw_output") or "")
        return {
            "passed": None if is_partial else d.get("passed"),
            "score": None if is_partial else d.get("score"),
            "eval_error": eval_error,
            "turns": len(calls),
            "duration": sum(c.get("duration_seconds") or 0 for c in calls),
            "kJ_clean": clean_J / 1000,
            "n_outliers": n_out,
            "input_tokens":  tu.get("input_tokens")  or sum(c.get("input_tokens")  or 0 for c in calls),
            "output_tokens": tu.get("output_tokens") or sum(c.get("output_tokens") or 0 for c in calls),
            "cached_tokens": sum(c.get("cache_h_delta") or 0 for c in calls),
            "patch_len": len(d.get("patch") or ""),
            "in_progress": is_partial,
            "phase": "eval-pending" if is_partial else None,
            **m,
        }
    except Exception:
        return {"passed": None, "score": None, "eval_error": None,
                "turns": 0, "duration": None,
                "kJ_clean": 0, "n_outliers": 0,
                "input_tokens": 0, "output_tokens": 0, "cached_tokens": 0,
                "patch_len": 0,
                "in_progress": False,
                "max_agent_ctx": None, "n_agents": 0,
                "n_reads_truncated": 0}


def _split_agent_seed(name: str) -> tuple[str, str]:
    if "_seed" not in name:
        return name, ""
    head, tail = name.rsplit("_seed", 1)
    parts = tail.split("_", 1)
    seed = parts[0]
    suffix = parts[1] if len(parts) > 1 else ""
    method = f"{head}_{suffix}" if suffix else head
    return method, seed


def _compute_summary(entries: list[dict]) -> dict:
    from collections import defaultdict
    per_method = defaultdict(list)
    per_task_method = defaultdict(list)
    for e in entries:
        method, seed = _split_agent_seed(e["agent_seed"])
        s = _episode_stats(Path(e["path"]))
        s["method"] = method
        s["task"] = e["task"]
        s["seed"] = seed
        per_method[method].append(s)
        per_task_method[(e["task"], method)].append(s)
    return {"per_method": dict(per_method), "per_task_method": dict(per_task_method)}


def _avg(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _fmt(v, prec=1):
    if v is None: return "—"
    if isinstance(v, bool): return "✓" if v else "✗"
    return f"{v:.{prec}f}" if isinstance(v, float) else str(v)


def _index_html(matrix_dir: Path, entries: list[dict]) -> str:
    from urllib.parse import quote
    import json as _json

    methods = sorted({_split_agent_seed(e["agent_seed"])[0] for e in entries})
    tasks = sorted({e["task"] for e in entries})

    tm_paths: dict[tuple[str, str], list[str]] = {}
    for e in entries:
        method, _ = _split_agent_seed(e["agent_seed"])
        tm_paths.setdefault((e["task"], method), []).append(e["path"])

    method_rows = []
    for m in methods:
        method_rows.append(
            f"<tr data-method='{html.escape(m)}'>"
            f"<td><b>{html.escape(m)}</b></td>"
            f"<td class='m-done'>—</td><td class='m-pass'>—</td>"
            f"<td class='m-turns'>—</td><td class='m-dur'>—</td>"
            f"<td class='m-kJcln'>—</td>"
            f"<td class='m-intok'>—</td>"
            f"<td class='m-outtok'>—</td>"
            f"<td class='m-cachedtok'>—</td>"
            f"<td class='m-nagents'>—</td>"
            f"<td class='m-agentctx'>—</td>"
            f"</tr>"
        )

    PER_METHOD_COLS = 9
    SUBHEADER = ("<th>pass</th><th>turns</th><th title='sum of per-call duration_seconds (GPU time)'>GPU dur</th>"
                 "<th title='Idle-baseline-corrected joules (raw NVML diff − sum(idle_W) × duration)'>kJ</th>"
                 "<th title='total input tokens summed across LLM calls'>in_tok</th>"
                 "<th title='total output tokens'>out_tok</th>"
                 "<th title='sum of per-call cache_h_delta (prefix-cache hit tokens)'>cached_tok</th>"
                 "<th>n_ag</th><th>agent_ctx</th>")
    headers = "<tr><th>task</th>" + "".join(
        f"<th colspan={PER_METHOD_COLS}>{html.escape(m)}</th>" for m in methods
    ) + "</tr>" + "<tr><th></th>" + "".join(
        SUBHEADER for _ in methods
    ) + "</tr>"
    task_rows = []
    for t in tasks:
        cells = [f"<td>{html.escape(t)}</td>"]
        for m in methods:
            paths = tm_paths.get((t, m), [])
            if not paths:
                cells.append(f"<td colspan={PER_METHOD_COLS} class='tm-cell missing'>—</td>")
            else:
                cells.append(
                    f"<td colspan={PER_METHOD_COLS} class='tm-cell loading' "
                    f"data-task='{html.escape(t)}' data-method='{html.escape(m)}'>…</td>"
                )
        task_rows.append("<tr>" + "".join(cells) + "</tr>")

    ep_rows = []
    for e in sorted(entries, key=lambda x: (x["task"], x["agent_seed"])):
        method, _ = _split_agent_seed(e["agent_seed"])
        href = f"/episode?path={quote(e['path'])}"
        init_status = "live" if e["in_progress"] else "pending"
        phase = e.get("phase") or ("agent-live" if e["in_progress"] else "")
        searchable = f"{e['task']} {e['agent_seed']} {method}"
        if e["in_progress"]:
            chip = (
                "<em style='color:#c80'>(agent)</em>"
                if phase == "agent-live"
                else "<em style='color:#06a'>(eval)</em>"
                if phase == "eval-pending"
                else "<em style='color:#c80'>(live)</em>"
            )
        else:
            chip = "<em style='color:#aaa'>…</em>"
        ep_rows.append(
            f"<li data-path='{html.escape(e['path'])}' "
            f"data-text='{html.escape(searchable.lower())}' "
            f"data-status='{init_status}' "
            f"data-phase='{html.escape(phase)}' "
            f"data-method='{html.escape(method)}'>"
            f"<a href='{href}'>{html.escape(e['task'])} / "
            f"{html.escape(e['agent_seed'])}</a>"
            f" <span class='ep-chip'>{chip}</span>"
            f" <span class='ep-meta' style='color:#999'></span>"
            f"</li>"
        )

    return f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'>
<title>matrix: {html.escape(matrix_dir.name)}</title>
<style>{CSS}
ul {{ list-style: none; padding-left: 0; }}
li {{ padding: 4px 0; }}
li a {{ font-family: monospace; }}
table.summary {{ border-collapse: collapse; margin: 12px 0; font-size: 13px; }}
table.summary th, table.summary td {{ border: 1px solid #ddd; padding: 4px 10px; text-align: left; }}
table.summary thead th {{ background: #f0f0f0; }}
table.summary tr:nth-child(even) {{ background: #fafafa; }}
table.summary td {{ font-family: ui-monospace, Menlo, monospace; }}
table.summary td.loading {{ color: #aaa; font-style: italic; text-align: center; }}
table.summary td.missing {{ color: #aaa; text-align: center; }}
h3 {{ margin-top: 24px; }}
.chip {{ padding:2px 8px; margin:0 2px; border:1px solid #ccc; background:#fff;
         border-radius:3px; font-size:12px; cursor:pointer; font-family:monospace; }}
.chip.active {{ background:#234; color:#fff; border-color:#234; }}
#ep-list li {{ font-size:12px }}
#progress {{ position:fixed; top:0; left:0; height:3px; background:#48c; z-index:99; transition:width 200ms ease-out; }}
</style></head>
<body>
<div id='progress' style='width:0%'></div>
<h2>matrix: {html.escape(str(matrix_dir))}</h2>
<p>{len(entries)} episodes  <span id='loaded-count' style='color:#888; font-size:12px'></span></p>

<h3>aggregate per method</h3>
<table class='summary'>
  <thead><tr>
    <th>method</th><th>done/total</th><th>pass rate</th>
    <th>avg turns</th><th title="avg sum of per-call duration_seconds (GPU time)">avg GPU dur</th>
    <th title="avg idle-baseline-corrected joules per episode (raw NVML diff minus sum(idle_W) × duration)">avg kJ</th>
    <th title="avg total input tokens per episode (summed across LLM calls)">avg in tok</th>
    <th title="avg total output tokens per episode">avg out tok</th>
    <th title="avg sum of per-call cache_h_delta (prefix-cache hit tokens) per episode">avg cached tok</th>
    <th title="# of distinct agent_ids that made ≥1 LLM call">avg # agents</th>
    <th title="max input_tokens across any agent">avg max agent ctx</th>
  </tr></thead>
  <tbody id='m-tbody'>{''.join(method_rows)}</tbody>
</table>

<h3>per-task (seed-averaged)</h3>
<table class='summary'>
  <thead>{headers}</thead>
  <tbody id='t-tbody'>{''.join(task_rows)}</tbody>
</table>

<h3>all episodes</h3>
<div id='ep-filter' style='margin-bottom:8px'>
  <input id='ep-search' type='text' placeholder='filter… (task name, agent, status)'
         style='padding:4px 8px; font-family:monospace; width:340px; font-size:13px'
         oninput='filterEps()'>
  <span style='margin-left:14px; font-size:12px'>status:</span>
  <button class='chip' data-status='all' onclick='setStatus(this)'>all</button>
  <button class='chip' data-status='pass' onclick='setStatus(this)'>pass</button>
  <button class='chip' data-status='fail' onclick='setStatus(this)'>fail</button>
  <button class='chip' data-status='eval_error' onclick='setStatus(this)' style='border-color:#b35;color:#b35'>eval_error</button>
  <button class='chip' data-status='live' onclick='setStatus(this)'>live</button>
  <span id='ep-count' style='margin-left:14px; font-size:12px; color:#666'></span>
</div>
<ul id='ep-list'>{''.join(ep_rows)}</ul>

<script>
// All stats fetched, indexed by path string. Filled by lazy loader.
const allStats = {{}};   // path → stats dict
let curStatus = 'all';

function setStatus(btn) {{
  curStatus = btn.dataset.status;
  document.querySelectorAll('#ep-filter .chip').forEach(c =>
    c.classList.toggle('active', c.dataset.status === curStatus));
  filterEps();
}}
function filterEps() {{
  const q = document.getElementById('ep-search').value.toLowerCase().trim();
  const tokens = q.split(/\\s+/).filter(Boolean);
  let visible = 0;
  document.querySelectorAll('#ep-list li').forEach(li => {{
    const text = li.dataset.text || '';
    const status = li.dataset.status || '';
    const okText = tokens.every(t => text.includes(t));
    const okStatus = (curStatus === 'all') || (status === curStatus);
    const show = okText && okStatus;
    li.style.display = show ? '' : 'none';
    if (show) visible++;
  }});
  document.getElementById('ep-count').textContent = `${{visible}} shown`;
}}

function fmt(v, prec) {{
  if (v === null || v === undefined) return '—';
  if (typeof v === 'boolean') return v ? '✓' : '✗';
  if (typeof v === 'number') return v.toFixed(prec);
  return String(v);
}}
function avg(xs) {{
  const ys = xs.filter(x => x !== null && x !== undefined);
  if (!ys.length) return null;
  return ys.reduce((a,b) => a+b, 0) / ys.length;
}}
// Mean only (CI intentionally omitted). Plain-text for HTML cells.
function avgCI(xs, prec) {{
  const ys = xs.filter(x => x !== null && x !== undefined);
  if (!ys.length) return '—';
  const m = ys.reduce((a,b) => a+b, 0) / ys.length;
  return fmt(m, prec);
}}
// Token-count formatter: 1234 → '1.2K', 1234567 → '1.2M'.
function fmtTok(n, prec) {{
  if (n === null || n === undefined) return '—';
  const ax = Math.abs(n);
  if (ax >= 1e6) return (n/1e6).toFixed(prec) + 'M';
  if (ax >= 1e3) return (n/1e3).toFixed(prec) + 'K';
  return Math.round(n).toString();
}}
function avgTok(xs, prec) {{
  const ys = xs.filter(x => x !== null && x !== undefined);
  if (!ys.length) return '—';
  return fmtTok(ys.reduce((a,b) => a+b, 0) / ys.length, prec);
}}

function updateAggregates() {{
  // Recompute from allStats. Walk every episode <li>.
  // Excluded from aggregates:
  //   - in-progress (live)
  //   - eval-side errors (s.eval_error set, e.g. harness_dns_error) —
  //     re-runnable via scripts/reevaluate.py; not a real agent outcome
  //   - zero-turn episodes (no LLM call made; usually crashed before run)
  // Empty-patch (but ≥1 turn, no eval error) episodes ARE included —
  // they count as fail in pass rate and contribute to mean turns / energy.
  const byMethod = {{}};        // method → list of all completed stats
  const byTaskMethod = {{}};    // 'task||method' → list
  const completedByMethod = {{}};  // method → count of completed episodes
  document.querySelectorAll('#ep-list li').forEach(li => {{
    const path = li.dataset.path;
    const method = li.dataset.method;
    const task = li.dataset.text.split(' ')[0];   // first token = task
    const s = allStats[path];
    if (!s) return;
    if (s.in_progress) return;   // skip live for aggregates
    if (s.eval_error) return;    // exclude eval-side harness failures
    if ((s.turns ?? 0) === 0) return;  // exclude zero-turn episodes
    completedByMethod[method] = (completedByMethod[method] || 0) + 1;
    (byMethod[method] = byMethod[method] || []).push(s);
    const tk = task + '||' + method;
    (byTaskMethod[tk] = byTaskMethod[tk] || []).push(s);
  }});
  // Per-method rows
  document.querySelectorAll('#m-tbody tr').forEach(tr => {{
    const m = tr.dataset.method;
    const ss = byMethod[m] || [];
    const totalRow = document.querySelectorAll(`#ep-list li[data-method="${{m}}"]`).length;
    if (!ss.length) {{
      tr.querySelector('.m-done').textContent = `0/${{totalRow}}`;
      return;
    }}
    const nPass = ss.filter(s => s.passed).length;
    tr.querySelector('.m-done').textContent  = `${{ss.length}}/${{totalRow}}`;
    tr.querySelector('.m-pass').textContent  = `${{nPass}}/${{ss.length}} (${{Math.round(100*nPass/ss.length)}}%)`;
    tr.querySelector('.m-turns').textContent = avgCI(ss.map(s=>s.turns), 0);
    tr.querySelector('.m-dur').textContent   = avgCI(ss.map(s=>s.duration), 0) + 's';
    tr.querySelector('.m-kJcln').textContent = avgCI(ss.map(s=>s.kJ_clean), 1);
    tr.querySelector('.m-intok').textContent  = avgTok(ss.map(s=>s.input_tokens), 1);
    tr.querySelector('.m-outtok').textContent = avgTok(ss.map(s=>s.output_tokens), 1);
    tr.querySelector('.m-cachedtok').textContent = avgTok(ss.map(s=>s.cached_tokens), 1);
    tr.querySelector('.m-nagents').textContent = avgCI(ss.map(s=>s.n_agents), 1);
    tr.querySelector('.m-agentctx').textContent = avgTok(ss.map(s=>s.max_agent_ctx), 1);
  }});
  // Per-task cells
  document.querySelectorAll('td.tm-cell.loading').forEach(td => {{
    const tk = td.dataset.task + '||' + td.dataset.method;
    const ss = byTaskMethod[tk];
    if (!ss || !ss.length) return;
    const nPass = ss.filter(s => s.passed).length;
    const passStr = ss.length > 1 ? `${{nPass}}/${{ss.length}}` : (nPass ? '✓' : '✗');
    td.colSpan = 1;
    td.classList.remove('loading');
    // Replace the single colspan cell with PER_METHOD_COLS cells in this row.
    const newCells = [
      `<td>${{passStr}}</td>`,
      `<td>${{avgCI(ss.map(s=>s.turns), 0)}}</td>`,
      `<td>${{avgCI(ss.map(s=>s.duration), 0)}}s</td>`,
      `<td>${{avgCI(ss.map(s=>s.kJ_clean), 1)}}</td>`,
      `<td>${{avgTok(ss.map(s=>s.input_tokens), 1)}}</td>`,
      `<td>${{avgTok(ss.map(s=>s.output_tokens), 1)}}</td>`,
      `<td>${{avgTok(ss.map(s=>s.cached_tokens), 1)}}</td>`,
      `<td>${{avgCI(ss.map(s=>s.n_agents), 1)}}</td>`,
      `<td>${{avgTok(ss.map(s=>s.max_agent_ctx), 1)}}</td>`,
    ].join('');
    td.outerHTML = newCells;
  }});
}}

function applyEpisodeStats(li, s) {{
  // Determine status chip. ``s.eval_error`` is set by
  // log_viewer's ``_detect_eval_error`` when the harness/eval crashed
  // (DNS, timeout, network) — distinct from a legit ``passed=False``.
  // Re-runnable via scripts/reevaluate.py.
  let status, chipHTML;
  if (s.in_progress) {{
    status = 'live';
    if (s.phase === 'eval-pending') {{
      chipHTML = "<em style='color:#06a'>(eval)</em>";
    }} else {{
      chipHTML = "<em style='color:#c80'>(agent)</em>";
    }}
  }} else if (s.eval_error) {{
    status = 'eval_error';
    chipHTML =
      "<span style='color:#b35; background:#fde; padding:1px 6px; "
      + "border:1px solid #b35; border-radius:3px;' "
      + `title='${{s.eval_error}} — re-run via scripts/reevaluate.py'>`
      + `⚠ eval_error: ${{s.eval_error}}</span>`;
  }} else if (s.passed === true) {{
    status = 'pass';
    chipHTML = "<span style='color:#080'>✓ pass</span>";
  }} else if (s.passed === false) {{
    status = 'fail';
    chipHTML = "<span style='color:#a00'>✗ fail</span>";
  }} else {{
    status = 'done';
    chipHTML = '';
  }}
  li.dataset.status = status;
  li.dataset.text = li.dataset.text + ' ' + status;
  li.querySelector('.ep-chip').innerHTML = chipHTML;
  const metaEl = li.querySelector('.ep-meta');
  metaEl.innerHTML =
    `· turns=${{s.turns ?? '?'}} kJ=${{fmt(s.kJ_clean, 1)}}` +
    ` in=${{fmtTok(s.input_tokens, 1)}} out=${{fmtTok(s.output_tokens, 1)}} cached=${{fmtTok(s.cached_tokens, 1)}}`;
}}

// Lazy load: fetch /episode_stats per <li> in parallel batches.
async function loadAll() {{
  const lis = Array.from(document.querySelectorAll('#ep-list li'));
  const total = lis.length;
  let done = 0;
  const BATCH = 16;      // concurrent /episode_stats fetches (server is ThreadingHTTPServer)
  let i = 0;
  async function worker() {{
    while (i < lis.length) {{
      const li = lis[i++];
      const path = li.dataset.path;
      try {{
        const r = await fetch('/episode_stats?path=' + encodeURIComponent(path));
        if (r.ok) {{
          const s = await r.json();
          allStats[path] = s;
          applyEpisodeStats(li, s);
        }}
      }} catch (e) {{ /* ignore */ }}
      done++;
      const pct = Math.round(100 * done / total);
      document.getElementById('progress').style.width = pct + '%';
      document.getElementById('loaded-count').textContent =
        `(${{done}}/${{total}} stats loaded)`;
    }}
  }}
  await Promise.all(Array.from({{length: BATCH}}, () => worker()));
  updateAggregates();
  filterEps();
  document.getElementById('progress').style.opacity = '0';
}}

// Init
document.querySelector('#ep-filter .chip[data-status=all]').classList.add('active');
filterEps();
loadAll();
</script>
</body></html>
"""


def _serve(matrix_dir: Path, port: int) -> None:
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlparse, parse_qs

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _send(self, body: str, code: int = 200, ctype: str = "text/html") -> None:
            data = body.encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            url = urlparse(self.path)
            try:
                if url.path == "/" or url.path == "/index":
                    entries = _list_episodes(matrix_dir)
                    self._send(_index_html(matrix_dir, entries))
                    return
                if url.path in ("/episode", "/episode_stats", "/turn"):
                    qs = parse_qs(url.query)
                    p = (qs.get("path") or [""])[0]
                    if not p:
                        self._send("missing ?path=", code=400)
                        return
                    pp = Path(os.path.normpath(p))
                    if not pp.is_absolute():
                        pp = Path(os.path.abspath(str(Path.cwd() / pp)))
                    md_abs = Path(os.path.abspath(str(matrix_dir)))
                    if not str(pp).startswith(str(md_abs) + "/"):
                        self._send("path outside matrix dir", code=403)
                        return
                    if not pp.exists():
                        self._send("not found", code=404)
                        return
                    if url.path == "/episode_stats":
                        s = _episode_stats(pp)
                        self._send(json.dumps(s, default=str), ctype="application/json")
                        return
                    if url.path == "/turn":
                        idx_str = (qs.get("idx") or ["-1"])[0]
                        try:
                            idx = int(idx_str)
                        except ValueError:
                            self._send("bad idx", code=400); return
                        turns = _get_episode_turns(pp)
                        if idx < 0 or idx >= len(turns):
                            self._send("idx out of range", code=404); return
                        self._send(_render_turn_body(turns[idx]))
                        return
                    meta = _load_episode(pp)
                    meta["_ep_path"] = str(pp)
                    turns = build_turns(meta.get("llm_calls") or [])
                    body = render_html(meta, turns)
                    body = body.replace(
                        "<body",
                        "<body",
                        1,
                    )
                    body = body.replace(
                        "<body data-ep-path=",
                        "<body><div style='margin-bottom:10px'><a href='/'>← back to index</a></div><div data-ep-path=",
                        1,
                    ) if False else body
                    import re as _re
                    body = _re.sub(
                        r"(<body[^>]*>)",
                        r"\1<div style='margin-bottom:10px'><a href='/'>← back to index</a></div>",
                        body, count=1,
                    )
                    self._send(body)
                    return
                self._send("404", code=404)
            except Exception as e:
                import traceback
                self._send(
                    f"<pre>{html.escape(traceback.format_exc())}</pre>",
                    code=500,
                )

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"[viewer] serving {matrix_dir} on http://localhost:{port}/")
    print("[viewer] Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[viewer] bye.")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--episode", type=Path,
                   help="Path to a *Evaluator*.json episode log "
                        "(static-HTML mode).")
    p.add_argument("--out", type=Path,
                   help="Output HTML path (static-HTML mode).")
    p.add_argument("--matrix-dir", type=Path,
                   help="Path to a matrix log dir, e.g. "
                        "logs/swebench_hot3_sharedfs_v2_3_seed0 "
                        "(live-server mode).")
    p.add_argument("--port", type=int, default=8765,
                   help="Port for live-server mode (default 8765).")
    args = p.parse_args()

    if args.matrix_dir:
        if not args.matrix_dir.is_dir():
            raise SystemExit(f"not a dir: {args.matrix_dir}")
        _serve(args.matrix_dir, args.port)
        return

    if not args.episode or not args.out:
        raise SystemExit(
            "either --matrix-dir (live mode) or "
            "--episode+--out (static mode) is required"
        )
    d = _load_episode(args.episode)
    turns = build_turns(d.get("llm_calls") or [])
    args.out.write_text(render_html(d, turns))
    print(f"wrote {args.out}  ({len(turns)} turns)")


if __name__ == "__main__":
    main()
