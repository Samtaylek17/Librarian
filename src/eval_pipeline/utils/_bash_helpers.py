"""Adapter re-exporting bashlex helpers from the vendored OpenHands legacy.

OpenHands V0's ``runtime/utils/bash.py`` carries two pure-function helpers
that we want to use verbatim:

- :func:`split_bash_commands` — bashlex AST split. We reject input that
  parses to >1 top-level command; this is the cheapest way to keep the
  PS1-sentinel detector sane (each tmux ``send-keys`` is a single
  logical statement → at most one extra PS1 emit per call).
- :func:`escape_bash_special_chars` — escapes ``\\;``, ``\\|``, ``\\&`` etc.
  in unquoted regions while leaving quoted strings, command
  substitutions, and heredoc bodies untouched.

We import both from ``third_party/openhands_legacy_vendored/bash.py``
through a small ``sys.path`` insertion. The vendored file is COMMIT_PIN'd
and checked by ``scripts/check_third_party_drift.py``.
"""
from __future__ import annotations

import sys
from pathlib import Path

_VENDORED_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "third_party"
    / "openhands_legacy_vendored"
)
if str(_VENDORED_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDORED_DIR))

# The vendored ``bash.py`` does ``from openhands.events.observation.commands
import bash as _legacy  # type: ignore[import-not-found]

split_bash_commands = _legacy.split_bash_commands
escape_bash_special_chars = _legacy.escape_bash_special_chars

__all__ = ["split_bash_commands", "escape_bash_special_chars"]
