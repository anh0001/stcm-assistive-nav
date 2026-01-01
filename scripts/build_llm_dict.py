#!/usr/bin/env python3
"""Build and/or update the LLM summary dict from STCM graph data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_PATH = REPO_ROOT / "output" / "stcm.json"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "output" / "llm_summary.json"

PACKAGE_ROOT = REPO_ROOT / "stcm"
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from stcm.map_utils import _build_llm_summary, _graph_data_to_graph, _is_stcm_payload


def _load_payload(path: Path) -> Dict[str, Any]:
    with path.open("r") as handle:
        return json.load(handle)


def _extract_graphs(
    data: Dict[str, Any]
) -> Tuple[Any, Optional[Any]]:
    if _is_stcm_payload(data):
        semantic_data = data.get("semantic_graph") or {}
        place_data = data.get("place_graph")
    else:
        semantic_data = data
        place_data = None
    semantic_graph = _graph_data_to_graph(semantic_data)
    place_graph = _graph_data_to_graph(place_data) if place_data is not None else None
    return semantic_graph, place_graph


def _save_json(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=4)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build the LLM summary dict from an STCM JSON payload."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=f"Path to STCM JSON (default: {DEFAULT_INPUT_PATH})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "Where to write the updated STCM payload. "
            "Defaults to overwriting the input STCM JSON when possible."
        ),
    )
    parser.add_argument(
        "--write-llm-summary",
        action="store_true",
        help="Also write the LLM summary to --llm-summary-output.",
    )
    parser.add_argument(
        "--llm-summary-output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"Path for the LLM summary JSON (default: {DEFAULT_OUTPUT_PATH})",
    )
    args = parser.parse_args()

    data = _load_payload(args.input)
    semantic_graph, place_graph = _extract_graphs(data)
    llm_summary = _build_llm_summary(semantic_graph, place_graph)
    summary_path = args.llm_summary_output if args.write_llm_summary else None
    if _is_stcm_payload(data):
        data["llm"] = llm_summary
        output_path = args.output or args.input
        if summary_path is not None:
            if summary_path.resolve() == output_path.resolve():
                raise SystemExit(
                    "--llm-summary-output must be different from the STCM output path."
                )
            _save_json(llm_summary, summary_path)
        _save_json(data, output_path)
    else:
        output_path = args.output or DEFAULT_OUTPUT_PATH
        _save_json(llm_summary, output_path)
        if summary_path is not None and summary_path.resolve() != output_path.resolve():
            _save_json(llm_summary, summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
