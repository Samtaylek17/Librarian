"""Reusable system-prompt sources.

Each module is one prompt or one prompt-template family. OH is one
source among many — nothing forces a runner to pick it.

  - :mod:`oh_system` — vendored OpenHands ``system_prompt.j2``
"""

from eval_pipeline.agents.prompts.oh_system import (
    _load_oh_system_prompt,
    load_oh_system_prompt,
)

__all__ = [
    "_load_oh_system_prompt",
    "load_oh_system_prompt",
]
