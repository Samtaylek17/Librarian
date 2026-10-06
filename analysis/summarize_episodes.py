"""Build comparison-friendly summaries from the episode-level CSV.

The method summary keeps experiment matrices separate, reports task/seed
coverage, and computes statistics both with and without execution-error rows.
The task comparison pivots methods side by side while preserving every raw
source field that is not part of the shared task identity.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Sequence


ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT = ROOT / "data" / "episodes.csv"
DEFAULT_METHOD_OUTPUT = ROOT / "tables" / "data" / "episodes_method_summary.csv"
DEFAULT_TASK_OUTPUT = ROOT / "tables" / "data" / "episodes_task_comparison.csv"

METHOD_GROUP_FIELDS = ("matrix_root", "benchmark", "model", "method", "agent")
CONTEXT_FIELDS = ("matrix_root", "benchmark", "model")
TASK_KEY_FIELDS = ("matrix_root", "benchmark", "model", "task_id", "seed")
REQUIRED_FIELDS = set(METHOD_GROUP_FIELDS) | {"task_id", "seed", "passed", "failure_mode"}
NON_STAT_FIELDS = set(METHOD_GROUP_FIELDS) | {"task_id", "seed", "passed", "failure_mode"}
STAT_NAMES = ("count", "sum", "mean", "median", "min", "max")


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"{path} has no CSV header")
        fieldnames = list(reader.fieldnames)
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError(f"{path} contains duplicate column names")
        missing = sorted(REQUIRED_FIELDS - set(fieldnames))
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(missing)}")
        rows = [dict(row) for row in reader]
    return fieldnames, rows


def _parse_bool(value: str | None) -> bool | None:
    normalized = (value or "").strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    return None


def _parse_number(value: str | None) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _numeric_fields(fieldnames: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    fields: list[str] = []
    for field in fieldnames:
        if field in NON_STAT_FIELDS:
            continue
        values = [(row.get(field) or "").strip() for row in rows]
        nonblank = [value for value in values if value]
        if nonblank and all(_parse_number(value) is not None for value in nonblank):
            fields.append(field)
    return fields


def _slug(value: str) -> str:
    slug = "".join(char.lower() if char.isalnum() else "_" for char in value)
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "unnamed"


def _failure_mode_columns(rows: Sequence[dict[str, str]]) -> list[tuple[str, str]]:
    modes = sorted({(row.get("failure_mode") or "").strip() for row in rows} - {""})
    columns = [(mode, f"failure_mode_{_slug(mode)}_count") for mode in modes]
    names = [column for _, column in columns]
    if len(names) != len(set(names)):
        raise ValueError("failure_mode values collapse to duplicate summary column names")
    return columns


def _round(number: float) -> float:
    return round(number, 6)


def _cohort_counts(rows: Sequence[dict[str, str]], prefix: str) -> dict[str, int | float | str]:
    outcomes = [_parse_bool(row.get("passed")) for row in rows]
    evaluated = [outcome for outcome in outcomes if outcome is not None]
    passed = sum(outcome is True for outcome in evaluated)
    failed = sum(outcome is False for outcome in evaluated)
    return {
        f"{prefix}_episode_count": len(rows),
        f"{prefix}_evaluated_count": len(evaluated),
        f"{prefix}_passed_count": passed,
        f"{prefix}_failed_count": failed,
        f"{prefix}_pass_rate_pct": _round(100.0 * passed / len(evaluated)) if evaluated else "",
    }


def _numeric_stats(
    rows: Sequence[dict[str, str]], numeric_fields: Sequence[str], prefix: str
) -> dict[str, int | float | str]:
    result: dict[str, int | float | str] = {}
    for field in numeric_fields:
        values = [
            number
            for row in rows
            if (number := _parse_number(row.get(field))) is not None
        ]
        base = f"{prefix}__{field}__"
        result[base + "count"] = len(values)
        if not values:
            for stat in STAT_NAMES[1:]:
                result[base + stat] = ""
            continue
        result[base + "sum"] = _round(sum(values))
        result[base + "mean"] = _round(statistics.fmean(values))
        result[base + "median"] = _round(statistics.median(values))
        result[base + "min"] = _round(min(values))
        result[base + "max"] = _round(max(values))
    return result


def build_method_summary(
    fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> tuple[list[str], list[dict[str, object]]]:
    """Return method-summary fieldnames and rows."""
    numeric_fields = _numeric_fields(fieldnames, rows)
    failure_columns = _failure_mode_columns(rows)

    expected_pairs: dict[tuple[str, ...], set[tuple[str, str]]] = defaultdict(set)
    groups: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        context = tuple(row.get(field, "") for field in CONTEXT_FIELDS)
        expected_pairs[context].add((row.get("task_id", ""), row.get("seed", "")))
        group = tuple(row.get(field, "") for field in METHOD_GROUP_FIELDS)
        groups[group].append(row)

    coverage_fields = [
        "episode_count",
        "task_count",
        "seed_count",
        "observed_task_seed_count",
        "expected_task_seed_count",
        "missing_run_count",
        "coverage_pct",
        "execution_error_count",
    ]
    cohort_fields = [
        f"{prefix}_{suffix}"
        for prefix in ("all", "no_error")
        for suffix in (
            "episode_count",
            "evaluated_count",
            "passed_count",
            "failed_count",
            "pass_rate_pct",
        )
    ]
    numeric_stat_fields = [
        f"{prefix}__{field}__{stat}"
        for prefix in ("all", "no_error")
        for field in numeric_fields
        for stat in STAT_NAMES
    ]
    output_fields = (
        list(METHOD_GROUP_FIELDS)
        + coverage_fields
        + cohort_fields
        + [column for _, column in failure_columns]
        + numeric_stat_fields
    )

    output_rows: list[dict[str, object]] = []
    for group, group_rows in sorted(groups.items()):
        record: dict[str, object] = dict(zip(METHOD_GROUP_FIELDS, group))
        context = group[: len(CONTEXT_FIELDS)]
        expected = expected_pairs[context]
        observed = {(row.get("task_id", ""), row.get("seed", "")) for row in group_rows}
        error_rows = [row for row in group_rows if (row.get("failure_mode") or "").strip()]
        no_error_rows = [row for row in group_rows if not (row.get("failure_mode") or "").strip()]

        record.update(
            {
                "episode_count": len(group_rows),
                "task_count": len({row.get("task_id", "") for row in group_rows}),
                "seed_count": len({row.get("seed", "") for row in group_rows}),
                "observed_task_seed_count": len(observed),
                "expected_task_seed_count": len(expected),
                "missing_run_count": len(expected - observed),
                "coverage_pct": _round(100.0 * len(observed) / len(expected)) if expected else "",
                "execution_error_count": len(error_rows),
            }
        )
        record.update(_cohort_counts(group_rows, "all"))
        record.update(_cohort_counts(no_error_rows, "no_error"))
        mode_counts = Counter((row.get("failure_mode") or "").strip() for row in group_rows)
        for mode, column in failure_columns:
            record[column] = mode_counts[mode]
        record.update(_numeric_stats(group_rows, numeric_fields, "all"))
        record.update(_numeric_stats(no_error_rows, numeric_fields, "no_error"))
        output_rows.append(record)

    return output_fields, output_rows


def build_task_comparison(
    fieldnames: Sequence[str], rows: Sequence[dict[str, str]]
) -> tuple[list[str], list[dict[str, str]]]:
    """Return a wide task-by-method comparison, rejecting ambiguous duplicates."""
    methods = sorted({(row.get("method") or "").strip() for row in rows})
    if "" in methods:
        raise ValueError("method must be nonblank for task comparison")
    value_fields = [field for field in fieldnames if field not in TASK_KEY_FIELDS]
    output_fields = list(TASK_KEY_FIELDS) + [
        f"{method}__{field}" for method in methods for field in value_fields
    ]

    pivoted: dict[tuple[str, ...], dict[str, str]] = {}
    seen: set[tuple[tuple[str, ...], str]] = set()
    for row in rows:
        key = tuple(row.get(field, "") for field in TASK_KEY_FIELDS)
        method = (row.get("method") or "").strip()
        identity = (key, method)
        if identity in seen:
            detail = ", ".join(f"{field}={value!r}" for field, value in zip(TASK_KEY_FIELDS, key))
            raise ValueError(f"duplicate episode for {detail}, method={method!r}")
        seen.add(identity)
        record = pivoted.setdefault(key, dict(zip(TASK_KEY_FIELDS, key)))
        for field in value_fields:
            record[f"{method}__{field}"] = row.get(field, "")

    return output_fields, [pivoted[key] for key in sorted(pivoted)]


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temp_path = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp_path, path)
    except BaseException:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
        raise


def generate(input_path: Path, method_output: Path, task_output: Path) -> tuple[int, int, int]:
    """Generate both outputs and return source, method, and task row counts."""
    fieldnames, rows = _read_csv(input_path)
    method_fields, method_rows = build_method_summary(fieldnames, rows)
    task_fields, task_rows = build_task_comparison(fieldnames, rows)
    _write_csv(method_output, method_fields, method_rows)
    _write_csv(task_output, task_fields, task_rows)
    return len(rows), len(method_rows), len(task_rows)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--method-output", type=Path, default=DEFAULT_METHOD_OUTPUT)
    parser.add_argument("--task-output", type=Path, default=DEFAULT_TASK_OUTPUT)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    source_count, method_count, task_count = generate(
        args.input, args.method_output, args.task_output
    )
    print(
        f"summarized {source_count} episodes into {method_count} method rows "
        f"and {task_count} task rows"
    )
    print(f"wrote {args.method_output}")
    print(f"wrote {args.task_output}")


if __name__ == "__main__":
    main()
