#!/usr/bin/env python3
"""Aggregate episodes to one row per method using the original CSV columns."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from decimal import Decimal
from pathlib import Path


SUM_FIELDS = {
    "score",
    "input_tokens",
    "output_tokens",
    "cached_tokens",
    "total_tokens",
    "total_joules",
    "total_joules_clean",
    "agent_duration_seconds",
    "total_duration_seconds",
    "n_llm_calls",
    "cache_h_tokens",
    "total_gpu_duration_seconds",
}
MAX_FIELDS = {"max_input_tokens"}


def _decimal_text(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text or "0"


def _sum(rows: list[dict[str, str]], field: str) -> str:
    values = [Decimal(row[field]) for row in rows if row[field].strip()]
    return _decimal_text(sum(values)) if values else ""


def _max(rows: list[dict[str, str]], field: str) -> str:
    values = [Decimal(row[field]) for row in rows if row[field].strip()]
    return _decimal_text(max(values)) if values else ""


def _constant_values(rows: list[dict[str, str]], field: str) -> str:
    values = sorted({row[field] for row in rows if row[field].strip()})
    return "|".join(values)


def aggregate(source: Path, destination: Path) -> tuple[int, int, int]:
    with source.open(newline="", encoding="utf-8-sig") as source_file:
        reader = csv.DictReader(source_file)
        fieldnames = list(reader.fieldnames or [])
        if "method" not in fieldnames:
            raise ValueError("Input CSV must contain a 'method' column")
        rows = list(reader)

    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["method"]].append(row)

    output_rows: list[dict[str, str]] = []
    for method in sorted(groups):
        method_rows = groups[method]
        output: dict[str, str] = {}
        for field in fieldnames:
            if field in SUM_FIELDS:
                output[field] = _sum(method_rows, field)
            elif field in MAX_FIELDS:
                output[field] = _max(method_rows, field)
            elif field == "task_id":
                output[field] = str(len({row[field] for row in method_rows}))
            elif field == "timestamp":
                timestamps = sorted(row[field] for row in method_rows if row[field].strip())
                output[field] = f"{timestamps[0]}..{timestamps[-1]}" if timestamps else ""
            elif field == "passed":
                output[field] = str(
                    sum(row[field].strip().lower() in {"true", "1"} for row in method_rows)
                )
            elif field in {"eval_error", "failure_mode"}:
                output[field] = str(sum(bool(row[field].strip()) for row in method_rows))
            else:
                output[field] = _constant_values(method_rows, field)
        output_rows.append(output)

    with destination.open("w", newline="", encoding="utf-8") as destination_file:
        writer = csv.DictWriter(destination_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    return len(rows), len(output_rows), len(fieldnames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    source_count, method_count, column_count = aggregate(args.source, args.destination)
    print(
        f"aggregated {source_count} episodes into {method_count} methods "
        f"with {column_count} columns"
    )


if __name__ == "__main__":
    main()
