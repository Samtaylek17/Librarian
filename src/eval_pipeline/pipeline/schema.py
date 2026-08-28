from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_tokens: int = 0
    theoretical_cached_tokens: int = 0
    total_tokens: int = 0


@dataclass
class EnergyResult:
    total_joules: float = 0.0
    n_calls_measured: int = 0
    model_name: str = ""
    total_joules_clean: float = 0.0
    n_outlier_calls: int = 0
    outlier_joules: float = 0.0
    outlier_threshold_seconds: float = 50.0
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class EpisodeData:
    task_id: str = ""
    benchmark: str = ""
    agent: str = ""
    model: str = ""
    agent_mode: str = ""
    seed: int = 0
    timestamp: str = ""

    patch: str | None = None
    agent_duration_seconds: float = 0.0
    agent_error: str | None = None

    score: float = 0.0
    passed: bool = False
    eval_raw_output: str = ""
    eval_metadata: dict[str, Any] = field(default_factory=dict)

    token_usage: TokenUsage | None = None
    llm_calls: list[dict[str, Any]] | None = None
    energy: EnergyResult | None = None

    total_duration_seconds: float = 0.0

    # Per-container resource caps in effect for this episode (agent + eval
    # containers). Recorded so results are interpretable/reproducible.
    resource_limits: dict[str, Any] | None = None

    agent_raw_output: Any = None

    def to_dict(self) -> dict[str, Any]:
        d = {}
        for k, v in asdict(self).items():
            if v is not None:
                d[k] = v
        return d

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "EpisodeData":
        with open(path) as f:
            data = json.load(f)
        if "token_usage" in data and data["token_usage"]:
            data["token_usage"] = TokenUsage(**data["token_usage"])
        if "energy" in data and data["energy"]:
            data["energy"] = EnergyResult(**data["energy"])
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
