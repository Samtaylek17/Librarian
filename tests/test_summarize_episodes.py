from __future__ import annotations

import csv
from pathlib import Path

import pytest

from analysis.summarize_episodes import generate


FIELDS = [
    "matrix_root",
    "agent_dir",
    "method",
    "task_id",
    "benchmark",
    "agent",
    "model",
    "seed",
    "timestamp",
    "score",
    "passed",
    "input_tokens",
    "eval_error",
    "failure_mode",
    "note",
]


def _row(method: str, task: str, **overrides: str) -> dict[str, str]:
    row = {
        "matrix_root": "matrix-1",
        "agent_dir": f"{method}_seed0",
        "method": method,
        "task_id": task,
        "benchmark": "Bench",
        "agent": "AgentA" if method == "method_a" else "AgentB",
        "model": "Model-1",
        "seed": "0",
        "timestamp": "2026-01-01T00:00:00",
        "score": "1",
        "passed": "True",
        "input_tokens": "10",
        "eval_error": "",
        "failure_mode": "",
        "note": f"note-{method}-{task}",
    }
    row.update(overrides)
    return row


def _write_input(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader.fieldnames or []), list(reader)


def _fixture_rows() -> list[dict[str, str]]:
    return [
        _row("method_a", "task-1", input_tokens="10", score="1", passed="True"),
        _row(
            "method_a",
            "task-2",
            input_tokens="100",
            score="0.5",
            passed="",
            failure_mode="stuck",
        ),
        _row("method_a", "task-3", input_tokens="30", score="0", passed="False"),
        _row(
            "method_b",
            "task-1",
            input_tokens="20",
            score="0",
            passed="False",
            failure_mode="timeout",
        ),
        _row(
            "method_b",
            "task-2",
            input_tokens="",
            score="1",
            passed="True",
            eval_error="grader warning",
        ),
    ]


def test_generate_method_summary_and_task_comparison(tmp_path: Path) -> None:
    source = tmp_path / "episodes.csv"
    method_output = tmp_path / "method.csv"
    task_output = tmp_path / "task.csv"
    _write_input(source, _fixture_rows())

    assert generate(source, method_output, task_output) == (5, 2, 3)

    method_fields, method_rows = _read(method_output)
    by_method = {row["method"]: row for row in method_rows}
    method_a = by_method["method_a"]
    method_b = by_method["method_b"]

    assert method_a["episode_count"] == "3"
    assert method_a["task_count"] == "3"
    assert method_a["expected_task_seed_count"] == "3"
    assert method_a["missing_run_count"] == "0"
    assert method_a["coverage_pct"] == "100.0"
    assert method_a["execution_error_count"] == "1"
    assert method_a["all_evaluated_count"] == "2"
    assert method_a["all_passed_count"] == "1"
    assert method_a["all_failed_count"] == "1"
    assert method_a["all_pass_rate_pct"] == "50.0"
    assert method_a["no_error_episode_count"] == "2"
    assert method_a["failure_mode_stuck_count"] == "1"
    assert method_a["failure_mode_timeout_count"] == "0"

    assert method_a["all__input_tokens__count"] == "3"
    assert method_a["all__input_tokens__sum"] == "140.0"
    assert method_a["all__input_tokens__mean"] == "46.666667"
    assert method_a["all__input_tokens__median"] == "30.0"
    assert method_a["all__input_tokens__min"] == "10.0"
    assert method_a["all__input_tokens__max"] == "100.0"
    assert method_a["no_error__input_tokens__count"] == "2"
    assert method_a["no_error__input_tokens__sum"] == "40.0"
    assert method_a["no_error__input_tokens__mean"] == "20.0"

    assert method_b["observed_task_seed_count"] == "2"
    assert method_b["missing_run_count"] == "1"
    assert method_b["coverage_pct"] == "66.666667"
    assert method_b["all__input_tokens__count"] == "1"
    assert method_b["no_error__input_tokens__count"] == "0"
    assert method_b["no_error__input_tokens__mean"] == ""
    assert "all__score__max" in method_fields

    task_fields, task_rows = _read(task_output)
    by_task = {row["task_id"]: row for row in task_rows}
    task_1 = by_task["task-1"]
    task_3 = by_task["task-3"]

    assert task_1["method_a__method"] == "method_a"
    assert task_1["method_b__method"] == "method_b"
    assert task_1["method_a__note"] == "note-method_a-task-1"
    assert task_1["method_b__failure_mode"] == "timeout"
    assert task_3["method_a__input_tokens"] == "30"
    assert task_3["method_b__input_tokens"] == ""
    assert task_3["method_b__note"] == ""
    assert "method_a__eval_error" in task_fields
    assert "method_b__timestamp" in task_fields


def test_duplicate_task_method_episode_is_rejected_before_outputs_are_written(
    tmp_path: Path,
) -> None:
    source = tmp_path / "episodes.csv"
    method_output = tmp_path / "method.csv"
    task_output = tmp_path / "task.csv"
    duplicate = _row("method_a", "task-1")
    _write_input(source, [duplicate, duplicate.copy()])

    with pytest.raises(ValueError, match="duplicate episode.*method='method_a'"):
        generate(source, method_output, task_output)

    assert not method_output.exists()
    assert not task_output.exists()
