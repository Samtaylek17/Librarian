from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RuntimeConfig:

    run_timeout: int = 7200
    max_turns: int = 1000
    max_nudges: int = 2
    max_observation_length: int = 16000
    bash_timeout: int = 1800
