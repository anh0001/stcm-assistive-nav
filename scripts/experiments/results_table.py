#!/usr/bin/env python3
"""Print experiment summary metrics as a Markdown table."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SUMMARY = REPO_ROOT / "results" / "eval" / "summary.csv"

DEFAULT_COLUMNS = [
    "run_id",
    "scenario",
    "variant",
    "precision_1m",
    "recall_1m",
    "f1_1m",
    "xy_error_mean_m",
    "xy_error_rmse_m",
    "benchmark_tp",
    "benchmark_fp",
    "benchmark_fn",
    "failure_flags",
]

HEADERS = {
    "run_id": "run_id",
    "scenario": "scenario",
    "variant": "variant",
    "precision_1m": "precision_1m",
    "recall_1m": "recall_1m",
    "f1_1m": "f1_1m",
    "xy_error_mean_m": "xy_error_mean_m",
    "xy_error_rmse_m": "xy_error_rmse_m",
    "benchmark_tp": "TP",
    "benchmark_fp": "FP",
    "benchmark_fn": "FN",
    "failure_flags": "failure_flags",
}

NUMERIC_COLUMNS = {
    "precision_1m",
    "recall_1m",
    "f1_1m",
    "xy_error_mean_m",
    "xy_error_rmse_m",
    "benchmark_tp",
    "benchmark_fp",
    "benchmark_fn",
}


def _load_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truthy(value: str | None) -> bool:
    return str(value or "").lower() in {"1", "true", "yes"}


def _select_rows(args: argparse.Namespace, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    selected = rows
    if args.run_id:
        selected = [row for row in selected if row.get("run_id") == args.run_id]
    if args.scenario:
        selected = [row for row in selected if row.get("scenario") == args.scenario]
    if args.variant:
        selected = [row for row in selected if row.get("variant") == args.variant]
    if args.executed_only:
        selected = [row for row in selected if _truthy(row.get("executed"))]
    if args.benchmark_only:
        selected = [row for row in selected if _truthy(row.get("benchmark_available"))]

    if args.all or args.run_id:
        return selected
    return selected[-1:] if selected else []


def _format_value(value: str, *, precision: int) -> str:
    if value == "":
        return ""
    try:
        number = float(value)
    except ValueError:
        return value
    if number.is_integer():
        return str(int(number))
    return f"{number:.{precision}f}".rstrip("0").rstrip(".")


def _markdown_table(rows: Iterable[dict[str, str]], columns: list[str], precision: int) -> str:
    headers = [HEADERS.get(column, column) for column in columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---:" if column in NUMERIC_COLUMNS else "---" for column in columns) + " |",
    ]
    for row in rows:
        cells = [
            _format_value(row.get(column, ""), precision=precision)
            if column in NUMERIC_COLUMNS
            else row.get(column, "")
            for column in columns
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY), help="Path to results/eval/summary.csv.")
    parser.add_argument("--run-id", help="Show one run id.")
    parser.add_argument("--scenario", help="Filter by scenario.")
    parser.add_argument("--variant", help="Filter by variant.")
    parser.add_argument("--all", action="store_true", help="Show all matching rows instead of the latest row.")
    parser.add_argument(
        "--include-failed",
        action="store_false",
        dest="benchmark_only",
        help="Include rows without benchmark metrics.",
    )
    parser.add_argument(
        "--include-dry-runs",
        action="store_false",
        dest="executed_only",
        help="Include dry-run rows.",
    )
    parser.add_argument(
        "--columns",
        default=",".join(DEFAULT_COLUMNS),
        help="Comma-separated summary.csv columns to show.",
    )
    parser.add_argument("--precision", type=int, default=4, help="Decimal places for numeric metrics.")
    parser.set_defaults(benchmark_only=True, executed_only=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary_path = Path(args.summary).expanduser()
    if not summary_path.is_absolute():
        summary_path = REPO_ROOT / summary_path
    rows = _load_rows(summary_path)
    selected = _select_rows(args, rows)
    columns = [column.strip() for column in args.columns.split(",") if column.strip()]
    print(_markdown_table(selected, columns, args.precision))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
