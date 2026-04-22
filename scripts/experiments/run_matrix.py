#!/usr/bin/env python3
"""Run or dry-run STCM experiment matrices from the experiment manifest."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "experiments" / "manifest.yaml"
RUNNER = REPO_ROOT / "scripts" / "experiments" / "run_experiment.py"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _split_csv(value: str, available: list[str]) -> list[str]:
    if value == "all":
        return available
    return [item.strip() for item in value.split(",") if item.strip()]


def _sensitivity_items(manifest: dict[str, Any], mode: str) -> list[str | None]:
    if mode == "none":
        return [None]
    sensitivity = manifest.get("sensitivity") or {}
    items = []
    for key, values in sensitivity.items():
        for value in values:
            items.append(f"{key}={value}")
    return items or [None]


def _build_command(args, scenario: str, variant: str, sensitivity: str | None) -> list[str]:
    command = [
        sys.executable,
        str(RUNNER),
        "--manifest",
        str(Path(args.manifest).expanduser()),
        "--scenario",
        scenario,
        "--variant",
        variant,
        "--timeout-sec",
        str(args.timeout_sec),
    ]
    if sensitivity:
        command.extend(["--sensitivity", sensitivity])
    if args.no_run:
        command.append("--no-run")
    if args.skip_bag_hash:
        command.append("--skip-bag-hash")
    return command


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--scenario", default="all", help="'all' or comma-separated scenario names.")
    parser.add_argument("--variant", default="full", help="'all' or comma-separated variant names.")
    parser.add_argument(
        "--sweep",
        choices=("none", "sensitivity"),
        default="none",
        help="Run one pass per manifest sensitivity value.",
    )
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--no-run", action="store_true")
    parser.add_argument("--skip-bag-hash", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = _load_yaml(manifest_path)
    scenarios = _split_csv(args.scenario, sorted(manifest.get("scenarios", {})))
    variants = _split_csv(args.variant, sorted(manifest.get("variants", {})))
    sensitivities = _sensitivity_items(manifest, args.sweep)

    failures = 0
    for scenario in scenarios:
        for variant in variants:
            for sensitivity in sensitivities:
                command = _build_command(args, scenario, variant, sensitivity)
                print(" ".join(command), flush=True)
                completed = subprocess.run(command, cwd=REPO_ROOT)
                if completed.returncode != 0:
                    failures += 1
                    if not args.no_run:
                        return completed.returncode
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

