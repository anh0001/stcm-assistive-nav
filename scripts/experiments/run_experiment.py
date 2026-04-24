#!/usr/bin/env python3
"""Run deterministic STCM offline experiments and capture reviewer evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import selectors
import shutil
import shlex
import signal
import subprocess
import sys
import time
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from benchmark_stcm_graph import evaluate_graphs, write_csv


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_ROOT / "configs" / "experiments" / "manifest.yaml"
COMPLETION_MARKER = "ROSBAG PROCESSING COMPLETE"
GRAPH_SAVE_MARKER = "STCM graph saved to:"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except subprocess.CalledProcessError:
        return "unknown"


def _git_dirty() -> bool:
    try:
        output = subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO_ROOT, text=True
        )
    except subprocess.CalledProcessError:
        return True
    return bool(output.strip())


def _sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path_text: str | None, *, hash_file: bool = True) -> dict[str, Any]:
    if not path_text:
        return {"path": path_text, "exists": False}
    path = Path(path_text).expanduser()
    record: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    if path.exists() and path.is_file():
        record["bytes"] = path.stat().st_size
        if hash_file:
            record["sha256"] = _sha256_file(path)
    return record


def _resolve_repo_path(path_text: str | None) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def _bag_metadata_path(bag_path: Path) -> Path:
    return bag_path / "metadata.yaml"


def _bag_info(bag_path: Path, *, hash_bag: bool) -> dict[str, Any]:
    metadata_path = _bag_metadata_path(bag_path)
    info = _load_yaml(metadata_path).get("rosbag2_bagfile_information", {})
    topics = {}
    for item in info.get("topics_with_message_count", []) or []:
        topic_meta = item.get("topic_metadata", {})
        name = topic_meta.get("name")
        if name:
            topics[name] = {
                "type": topic_meta.get("type"),
                "message_count": item.get("message_count", 0),
            }
    files = []
    for rel_path in info.get("relative_file_paths", []) or []:
        db_path = bag_path / rel_path
        files.append(
            {
                "path": str(db_path),
                "bytes": db_path.stat().st_size if db_path.exists() else None,
                "sha256": _sha256_file(db_path) if hash_bag and db_path.exists() else None,
            }
        )
    return {
        "path": str(bag_path),
        "metadata_sha256": _sha256_file(metadata_path),
        "storage_identifier": info.get("storage_identifier"),
        "duration_ns": (info.get("duration") or {}).get("nanoseconds"),
        "message_count": info.get("message_count"),
        "topics": topics,
        "files": files,
    }


def _preflight_bag(bag_path: Path, required_topics: list[str]) -> list[str]:
    errors = []
    if not bag_path.exists():
        return [f"Bag path does not exist: {bag_path}"]
    metadata_path = _bag_metadata_path(bag_path)
    if not metadata_path.exists():
        return [f"Missing rosbag metadata: {metadata_path}"]
    info = _bag_info(bag_path, hash_bag=False)
    available = set(info["topics"])
    missing = [topic for topic in required_topics if topic not in available]
    if missing:
        errors.append(f"Missing required topics in {bag_path}: {', '.join(missing)}")
    return errors


def _graph_section(data: dict[str, Any], key: str) -> dict[str, Any]:
    if "semantic_graph" in data or "place_graph" in data:
        return data.get(key) or {"nodes": [], "links": []}
    if key == "semantic_graph":
        return data
    return {"nodes": [], "links": []}


def _label_distribution(nodes: list[dict[str, Any]]) -> dict[str, int]:
    distribution: dict[str, int] = {}
    for node in nodes:
        label = str(node.get("category") or node.get("label") or "unknown")
        distribution[label] = distribution.get(label, 0) + 1
    return dict(sorted(distribution.items()))


def _pose(node: dict[str, Any]) -> list[float] | None:
    pose = node.get("pose")
    if not isinstance(pose, list) or len(pose) < 2:
        return None
    try:
        return [float(pose[0]), float(pose[1])]
    except (TypeError, ValueError):
        return None


def _duplicate_indicators(nodes: list[dict[str, Any]], threshold_m: float = 0.25) -> dict[str, Any]:
    by_label: dict[str, list[tuple[str, list[float]]]] = {}
    for index, node in enumerate(nodes):
        pose = _pose(node)
        if pose is None:
            continue
        label = str(node.get("category") or node.get("label") or "unknown")
        node_id = str(node.get("id") or node.get("instance_id") or f"node_{index}")
        by_label.setdefault(label, []).append((node_id, pose))

    close_pairs = []
    min_distance_by_label: dict[str, float | None] = {}
    for label, entries in sorted(by_label.items()):
        min_distance = None
        for idx, (node_a, pose_a) in enumerate(entries):
            for node_b, pose_b in entries[idx + 1 :]:
                distance = ((pose_a[0] - pose_b[0]) ** 2 + (pose_a[1] - pose_b[1]) ** 2) ** 0.5
                if min_distance is None or distance < min_distance:
                    min_distance = distance
                if distance < threshold_m:
                    close_pairs.append(
                        {
                            "label": label,
                            "node_a": node_a,
                            "node_b": node_b,
                            "distance_m": distance,
                        }
                    )
        min_distance_by_label[label] = min_distance

    return {
        "threshold_m": threshold_m,
        "close_pair_count": len(close_pairs),
        "close_pairs": close_pairs[:50],
        "min_distance_by_label": min_distance_by_label,
    }


def _graph_metrics(graph_path: Path) -> dict[str, Any]:
    if not graph_path.exists():
        return {"exists": False, "path": str(graph_path)}
    with graph_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    semantic = _graph_section(data, "semantic_graph")
    place = _graph_section(data, "place_graph")
    object_nodes = semantic.get("nodes", []) or []
    place_nodes = place.get("nodes", []) or []
    metadata = data.get("metadata", {}) if isinstance(data, dict) else {}
    return {
        "exists": True,
        "path": str(graph_path),
        "sha256": _sha256_file(graph_path),
        "object_node_count": len(object_nodes),
        "object_edge_count": len(semantic.get("links", []) or semantic.get("edges", []) or []),
        "place_node_count": len(place_nodes),
        "place_edge_count": len(place.get("links", []) or place.get("edges", []) or []),
        "label_distribution": _label_distribution(object_nodes),
        "duplicate_indicators": _duplicate_indicators(object_nodes),
        "runtime": metadata.get("runtime", {}),
    }


def _parse_log_metrics(log_path: Path) -> dict[str, Any]:
    if not log_path.exists():
        return {}
    text = log_path.read_text(encoding="utf-8", errors="replace")
    return {
        "completion_marker_seen": COMPLETION_MARKER in text,
        "tf_lookup_failed_count": text.count("TF lookup failed"),
        "pose_failure_count": text.count("Failed to calculate 3D position"),
        "projected_lidar_fallback_count": text.count("falling back to depth"),
        "gng_pause_timeout_count": text.count("GNG pause timeout"),
        "warning_count": text.count("[WARN]") + text.count(" WARNING "),
        "error_count": text.count("[ERROR]") + text.count(" ERROR "),
    }


def _failure_flags(result: dict[str, Any]) -> list[str]:
    flags = []
    graph = result.get("graph", {})
    if not graph.get("exists"):
        flags.append("graph_missing")
    if graph.get("object_node_count", 0) == 0:
        flags.append("zero_object_nodes")
    runtime = graph.get("runtime", {})
    events = runtime.get("events", {}) if isinstance(runtime, dict) else {}
    frames_seen = int(events.get("frames_seen") or 0)
    zero_frames = int(events.get("zero_detection_frames") or 0)
    if frames_seen > 0 and zero_frames / frames_seen > 0.10:
        flags.append("zero_detection_frames_gt_10pct")
    if int(events.get("tf_lookup_failures") or 0) > 0:
        flags.append("tf_lookup_failures_present")
    if int(events.get("gng_update_failures") or 0) > 0:
        flags.append("gng_update_failures_present")
    if not result.get("log", {}).get("completion_marker_seen") and result.get("executed"):
        flags.append("completion_marker_missing")
    if result.get("launch", {}).get("timed_out"):
        flags.append("launch_timeout")
    benchmark = result.get("benchmark") or {}
    if result.get("executed") and benchmark.get("required") and not benchmark.get("available"):
        flags.append("benchmark_missing")
    return flags


def _make_run_id(scenario: str, variant: str, suffix: str | None) -> str:
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    parts = [timestamp, scenario, variant]
    if suffix:
        parts.append(suffix)
    return "_".join(part.replace("/", "-") for part in parts)


def _build_launch_command(
    config_path: Path,
    graph_path: Path,
    bag_path: Path,
    storage_id: str,
    ros_log_dir: Path,
    mpl_config_dir: Path,
) -> list[str]:
    setup_install = REPO_ROOT / "install" / "setup.bash"
    setup_parts = [
        'export PYTHONUSERBASE="${PYTHONUSERBASE:-$HOME/.local/stcm_sys_py310}"',
        f"export ROS_LOG_DIR={shlex.quote(str(ros_log_dir))}",
        f"export MPLCONFIGDIR={shlex.quote(str(mpl_config_dir))}",
        "source /opt/ros/humble/setup.bash",
    ]
    if setup_install.exists():
        setup_parts.append(f"source {setup_install}")
    setup_parts.append(
        "ros2 launch stcm semantic_mapping.launch.py "
        f"config_file:={config_path} "
        "offline_sequential:=true "
        "use_sim_time:=true "
        "run_updater:=false "
        f"rosbag_path:={bag_path} "
        f"rosbag_storage_id:={storage_id} "
        f"graph_output_path:={graph_path} "
        f"place_gng_output_path:={graph_path}"
    )
    return ["bash", "-lc", " && ".join(setup_parts)]


def _run_launch(command: list[str], log_path: Path, timeout_sec: int) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completion_seen = False
    graph_saved_seen = False
    timed_out = False
    returncode = None
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            preexec_fn=os.setsid,
        )
        try:
            assert process.stdout is not None
            selector = selectors.DefaultSelector()
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                if timeout_sec > 0 and time.perf_counter() - start > timeout_sec:
                    timed_out = True
                    os.killpg(process.pid, signal.SIGINT)
                    break

                events = selector.select(timeout=0.5)
                if not events:
                    if process.poll() is not None:
                        break
                    continue

                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue

                log_handle.write(line)
                log_handle.flush()
                if COMPLETION_MARKER in line:
                    completion_seen = True
                if completion_seen and GRAPH_SAVE_MARKER in line:
                    graph_saved_seen = True
                if graph_saved_seen:
                    time.sleep(2.0)
                    os.killpg(process.pid, signal.SIGINT)
                    break
            try:
                returncode = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGTERM)
                returncode = process.wait(timeout=10)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                returncode = process.wait(timeout=10)
    return {
        "returncode": returncode,
        "elapsed_sec": time.perf_counter() - start,
        "completion_seen": completion_seen,
        "graph_saved_seen": graph_saved_seen,
        "timed_out": timed_out,
    }


def _materialize_run_config(
    *,
    base_config: Path,
    overlay_path: Path,
    scenario: dict[str, Any],
    graph_path: Path,
    storage_id: str,
) -> dict[str, Any]:
    config = _load_yaml(base_config)
    overlay = _load_yaml(overlay_path)
    config = _deep_merge(config, overlay)
    config.update(
        {
            "offline_sequential": True,
            "use_sim_time": True,
            "run_updater": False,
            "rosbag_path": scenario["bag_path"],
            "rosbag_storage_id": storage_id,
            "graph_output_path": str(graph_path),
            "place_gng_output_path": str(graph_path),
        }
    )
    scenario_overrides = scenario.get("config_overrides") or {}
    if not isinstance(scenario_overrides, dict):
        raise TypeError("scenario config_overrides must be a mapping")
    config = _deep_merge(config, scenario_overrides)
    return config


def _apply_sensitivity(config: dict[str, Any], sensitivity: str | None) -> str | None:
    if not sensitivity:
        return None
    if "=" not in sensitivity:
        raise ValueError("--sensitivity must use key=value")
    key, raw_value = sensitivity.split("=", 1)
    integer_keys = {
        "gng_min_observations_to_commit",
        "instance_label_switch_min_observations",
    }

    def _coerce_value(name: str, value: str) -> Any:
        if name in integer_keys:
            return int(value)
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        return float(value)

    def _assign_nested(target: dict[str, Any], dotted_key: str, value: Any) -> None:
        if "." not in dotted_key:
            target[dotted_key] = value
            return
        cursor = target
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            next_cursor = cursor.get(part)
            if not isinstance(next_cursor, dict):
                next_cursor = {}
                cursor[part] = next_cursor
            cursor = next_cursor
        cursor[parts[-1]] = value

    if key == "box_text_threshold":
        value = float(raw_value)
        config["box_threshold"] = value
        config["text_threshold"] = value
    else:
        _assign_nested(config, key, _coerce_value(key, raw_value))
    key_token = key.replace(".", "_")
    return f"{key_token}-{raw_value}".replace(".", "p")


def _benchmark_metrics(
    *,
    scenario: dict[str, Any],
    graph_path: Path,
    artifact_dir: Path,
) -> dict[str, Any]:
    ground_truth_path = _resolve_repo_path(scenario.get("ground_truth_path"))
    if ground_truth_path is None:
        return {"required": False, "available": False}

    benchmark_json_path = artifact_dir / "benchmark.json"
    benchmark_csv_path = artifact_dir / "benchmark.csv"
    threshold_m = float(scenario.get("benchmark_match_threshold_m", 1.0))
    record: dict[str, Any] = {
        "required": True,
        "available": False,
        "ground_truth_path": str(ground_truth_path),
        "match_threshold_m": threshold_m,
        "output_json": str(benchmark_json_path),
        "output_csv": str(benchmark_csv_path),
    }
    if not ground_truth_path.exists():
        record["error"] = f"Ground truth path does not exist: {ground_truth_path}"
        return record
    if not graph_path.exists():
        record["error"] = f"Prediction graph path does not exist: {graph_path}"
        return record

    try:
        benchmark = evaluate_graphs(
            prediction_path=graph_path,
            ground_truth_path=ground_truth_path,
            match_threshold_m=threshold_m,
            label_aliases=scenario.get("benchmark_label_aliases"),
            composite_covers=scenario.get("benchmark_composite_covers"),
        )
    except Exception as exc:  # noqa: BLE001 - preserve failure evidence in result JSON.
        record["error"] = f"{type(exc).__name__}: {exc}"
        return record

    _write_json(benchmark_json_path, benchmark)
    write_csv(benchmark_csv_path, benchmark)
    record.update(
        {
            "available": True,
            "metric_name": benchmark.get("metric_name"),
            "match_policy": benchmark.get("match_policy", {}),
            "summary": benchmark.get("summary", {}),
            "per_label": benchmark.get("per_label", {}),
            "matched_pairs": benchmark.get("matched_pairs", []),
            "false_positive_nodes": benchmark.get("false_positive_nodes", []),
            "covered_false_positive_nodes": benchmark.get("covered_false_positive_nodes", []),
            "false_negative_gt_nodes": benchmark.get("false_negative_gt_nodes", []),
            "wrong_label_near_gt": benchmark.get("wrong_label_near_gt", []),
            "duplicate_pairs": benchmark.get("duplicate_pairs", []),
        }
    )
    return record


def _run_one(args, manifest: dict[str, Any], scenario_name: str, variant_name: str) -> Path:
    scenarios = manifest.get("scenarios", {})
    variants = manifest.get("variants", {})
    if scenario_name not in scenarios:
        raise KeyError(f"Unknown scenario: {scenario_name}")
    if variant_name not in variants:
        raise KeyError(f"Unknown variant: {variant_name}")

    scenario = dict(scenarios[scenario_name])
    bag_path = Path(scenario["bag_path"]).expanduser()
    storage_id = scenario.get("storage_id", "sqlite3")
    errors = _preflight_bag(bag_path, manifest.get("required_topics", []))
    if errors:
        raise RuntimeError("; ".join(errors))

    base_config = (REPO_ROOT / manifest.get("base_config", "")).resolve()
    overlay_path = (REPO_ROOT / variants[variant_name]["overlay"]).resolve()
    results_dir = (REPO_ROOT / manifest.get("results_dir", "results/eval")).resolve()
    run_id = _make_run_id(scenario_name, variant_name, None)
    artifact_dir = results_dir / "artifacts" / run_id
    graph_path = artifact_dir / "stcm.json"
    config_path = artifact_dir / "config.yaml"
    log_path = artifact_dir / "launch.log"
    result_path = results_dir / f"{run_id}.json"
    ros_log_dir = artifact_dir / "ros_logs"
    mpl_config_dir = artifact_dir / "mplconfig"

    config = _materialize_run_config(
        base_config=base_config,
        overlay_path=overlay_path,
        scenario=scenario,
        graph_path=graph_path,
        storage_id=storage_id,
    )
    sensitivity_suffix = _apply_sensitivity(config, args.sensitivity)
    if sensitivity_suffix:
        run_id = _make_run_id(scenario_name, variant_name, sensitivity_suffix)
        artifact_dir = results_dir / "artifacts" / run_id
        graph_path = artifact_dir / "stcm.json"
        config_path = artifact_dir / "config.yaml"
        log_path = artifact_dir / "launch.log"
        result_path = results_dir / f"{run_id}.json"
        ros_log_dir = artifact_dir / "ros_logs"
        mpl_config_dir = artifact_dir / "mplconfig"
        config["graph_output_path"] = str(graph_path)
        config["place_gng_output_path"] = str(graph_path)

    _write_yaml(config_path, config)
    ros_log_dir.mkdir(parents=True, exist_ok=True)
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    command = _build_launch_command(
        config_path,
        graph_path,
        bag_path,
        storage_id,
        ros_log_dir,
        mpl_config_dir,
    )
    command_text = shlex.join(command)

    launch_result = {
        "returncode": None,
        "elapsed_sec": None,
        "completion_seen": False,
        "skipped": bool(args.no_run),
    }
    if args.no_run:
        print(command_text)
    else:
        launch_result.update(_run_launch(command, log_path, args.timeout_sec))

    if graph_path.exists():
        copied_graph = artifact_dir / "graph_copy_for_evidence.json"
        if copied_graph != graph_path:
            shutil.copy2(graph_path, copied_graph)

    bag_info = _bag_info(bag_path, hash_bag=not args.skip_bag_hash)
    result = {
        "run_id": run_id,
        "reviewer_items": ["AE-1/R1-1", "AE-3/R1-3", "AE-4/R1-4", "AE-9/R2-4"],
        "scenario": scenario_name,
        "scenario_description": scenario.get("description"),
        "variant": variant_name,
        "sensitivity": args.sensitivity,
        "executed": not args.no_run,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git": {"sha": _git_sha(), "dirty": _git_dirty()},
        "bag": bag_info,
        "config": {
            "base": str(base_config),
            "overlay": str(overlay_path),
            "snapshot": str(config_path),
            "sha256": _sha256_file(config_path),
        },
        "checkpoints": {
            "groundingdino": _file_record(config.get("groundingdino_checkpoint")),
            "mobilesam": _file_record(config.get("mobilesam_checkpoint")),
            "depth_anything": _file_record(config.get("depth_anything_checkpoint"), hash_file=False),
        },
        "command": command,
        "command_text": command_text,
        "launch": launch_result,
        "log": _parse_log_metrics(log_path),
        "graph": _graph_metrics(graph_path),
        "benchmark": _benchmark_metrics(
            scenario=scenario,
            graph_path=graph_path,
            artifact_dir=artifact_dir,
        ),
    }
    result["failure_flags"] = _failure_flags(result)
    _write_json(result_path, result)
    _write_json(artifact_dir / "result.json", result)
    print(f"Wrote {result_path.relative_to(REPO_ROOT)}")
    return result_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--scenario", default="all", help="Scenario name or 'all'.")
    parser.add_argument("--variant", default="full", help="Variant name from manifest.")
    parser.add_argument("--sensitivity", help="Optional sweep override as key=value.")
    parser.add_argument("--timeout-sec", type=int, default=7200)
    parser.add_argument("--no-run", action="store_true", help="Materialize config and print command only.")
    parser.add_argument(
        "--skip-bag-hash",
        action="store_true",
        help="Record bag file sizes but skip db3 sha256 hashing.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_absolute():
        manifest_path = REPO_ROOT / manifest_path
    manifest = _load_yaml(manifest_path)
    scenarios = sorted(manifest.get("scenarios", {})) if args.scenario == "all" else [args.scenario]
    for scenario in scenarios:
        _run_one(args, manifest, scenario, args.variant)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
