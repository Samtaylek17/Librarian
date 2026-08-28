from __future__ import annotations

import json as _json
import re as _re


_VIEW_ACTION_RX = _re.compile(
    r"^\s*(?:cat|less|more)\b(?:\s+-n)?\s+(?P<path>[\'\"]?\/[\w\-./]+\b[\'\"]?)"
    r"|^\s*sed\b[^|;]*?-n\s+[\'\"](?P<sa>\d+)(?:,(?P<sb>\d+))?p[\'\"]\s+(?P<spath>[\'\"]?\/[\w\-./]+\b[\'\"]?)"
    r"|^\s*(?P<ht>head|tail)\b[^|;]*?-n\s+(?P<n>\d+)\s+(?P<htpath>[\'\"]?\/[\w\-./]+\b[\'\"]?)"
    r"|^\s*awk\b[^|;]*?NR\s*>=\s*(?P<aa>\d+)[^|;]*?NR\s*<=\s*(?P<ab>\d+)[^|;]*?(?P<apath>[\'\"]?\/[\w\-./]+\b[\'\"]?)",
    _re.M,
)


def parse_view_action(
    name: str, args_dict: dict
) -> tuple[str, int | None, int | None] | None:
    if name == "str_replace_editor":
        if args_dict.get("command") != "view":
            return None
        path = args_dict.get("path") or ""
        if not path:
            return None
        vr = args_dict.get("view_range")
        if isinstance(vr, list) and len(vr) == 2:
            try:
                a, b = int(vr[0]), int(vr[1])
                return (path, a, (None if b == -1 else b))
            except Exception:
                return (path, 1, None)
        return (path, 1, None)
    if name == "bash":
        cmd = (args_dict.get("command") or "").strip()
        if not cmd:
            return None
        m = _VIEW_ACTION_RX.search(cmd)
        if not m:
            return None
        if m.group("path"):
            return (m.group("path").strip("'\""), 1, None)
        if m.group("spath"):
            sa = int(m.group("sa"))
            sb = int(m.group("sb")) if m.group("sb") else sa
            return (m.group("spath").strip("'\""), sa, sb)
        if m.group("htpath"):
            n = int(m.group("n"))
            if m.group("ht") == "head":
                return (m.group("htpath").strip("'\""), 1, n)
            return (m.group("htpath").strip("'\""), 1, None)
        if m.group("apath"):
            return (
                m.group("apath").strip("'\""),
                int(m.group("aa")),
                int(m.group("ab")),
            )
    return None


class ViewedLinesTracker:

    def __init__(self) -> None:
        self._per_file: dict[str, set[int]] = {}
        self._fully_viewed: set[str] = set()

    def reset_file(self, path: str) -> None:
        self._per_file.pop(path, None)
        self._fully_viewed.discard(path)

    def measure_new(
        self,
        path: str,
        start: int | None,
        end: int | None,
        obs_chars: int,
    ) -> int:
        if not path or obs_chars <= 0:
            return 0
        if start is None or end is None:
            if path in self._fully_viewed:
                return 0
            self._fully_viewed.add(path)
            return obs_chars
        if end < start:
            return 0
        requested = set(range(start, end + 1))
        seen = self._per_file.get(path, set())
        new = requested - seen
        self._per_file.setdefault(path, set()).update(requested)
        if not new:
            return 0
        frac = len(new) / max(len(requested), 1)
        return int(round(frac * obs_chars))

    def snapshot(self) -> tuple[dict, set]:
        return (
            {k: set(v) for k, v in self._per_file.items()},
            set(self._fully_viewed),
        )

    def restore(self, snap: tuple[dict, set]) -> None:
        self._per_file, self._fully_viewed = snap[0], snap[1]

    def touched_files(self) -> set[str]:
        return set(self._per_file.keys()) | set(self._fully_viewed)


def measure_session_new_chars(
    delta: list, tracker: ViewedLinesTracker
) -> int:
    outputs_by_id: dict[str, str] = {}
    outputs_in_order: list[str] = []
    calls_in_order: list[tuple[str, dict, str | None]] = []
    for msg in delta:
        if not isinstance(msg, dict):
            continue
        t = msg.get("type")
        if t == "function_call_output":
            out_raw = msg.get("output", "")
            out_text = str(out_raw) if out_raw is not None else ""
            outputs_in_order.append(out_text)
            cid = msg.get("call_id")
            if cid:
                outputs_by_id[cid] = out_text
        elif t == "function_call":
            try:
                args_dict = _json.loads(msg.get("arguments", "{}"))
            except Exception:
                args_dict = {}
            calls_in_order.append(
                (msg.get("name", ""), args_dict, msg.get("call_id"))
            )
    total_new = 0
    for idx, (name, args_dict, cid) in enumerate(calls_in_order):
        view = parse_view_action(name, args_dict)
        if view is None:
            continue
        path, a, b = view
        out_text = ""
        if cid and cid in outputs_by_id:
            out_text = outputs_by_id[cid]
        elif idx < len(outputs_in_order):
            out_text = outputs_in_order[idx]
        total_new += tracker.measure_new(path, a, b, len(out_text))
    return total_new
