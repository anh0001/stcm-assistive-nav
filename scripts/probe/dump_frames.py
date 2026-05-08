#!/usr/bin/env python3
"""Sample frames from a rosbag and save as PNG with robot-pose sidecar.

Used by the GDINO label-recognition probe to extract images where the robot
was near each GT object so we can ask: "is this label detectable here?"

Usage (basic, every Nth frame):
    python3 scripts/probe/dump_frames.py \
        --bag /mnt/STREAM/outdoor_livinglab_01_20260501_162329_0 \
        --image-topic /camera/image_raw \
        --output-dir scripts/probe/frames/outdoor \
        --every-n 5

Usage (filter to frames where robot is within `dist_m` of any GT pose):
    python3 scripts/probe/dump_frames.py \
        --bag /mnt/STREAM/outdoor_livinglab_01_20260501_162329_0 \
        --image-topic /camera/image_raw \
        --output-dir scripts/probe/frames/outdoor \
        --gt-json configs/experiments/ground_truth/outdoor_livinglab_stcm_gt.json \
        --max-dist-m 6.0 \
        --tf-base-frame base_footprint --tf-world-frame map

Output:
    <output-dir>/frame_<idx>.png       — RGB image
    <output-dir>/frame_<idx>.json      — {timestamp_ns, robot_xy or null, dist_to_nearest_gt_m or null}
    <output-dir>/manifest.json         — {bag, image_topic, every_n, frame_count, ...}
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _load_gt_xy(gt_json_path: Path) -> list[tuple[str, list[float]]]:
    if not gt_json_path or not gt_json_path.exists():
        return []
    data = json.loads(gt_json_path.read_text())
    nodes = data.get("semantic_graph", {}).get("nodes", [])
    out = []
    for n in nodes:
        pose = n.get("pose")
        label = n.get("category") or n.get("label") or n.get("id", "?")
        if isinstance(pose, list) and len(pose) >= 2:
            out.append((str(label), [float(pose[0]), float(pose[1])]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bag", required=True, help="Rosbag2 directory (sqlite3 or mcap)")
    ap.add_argument("--image-topic", default="/camera/image_raw")
    ap.add_argument("--tf-topic", default="/tf")
    ap.add_argument("--tf-static-topic", default="/tf_static")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--every-n", type=int, default=5,
                    help="Save every Nth image (default 5)")
    ap.add_argument("--max-frames", type=int, default=200,
                    help="Hard cap on number of frames written")
    ap.add_argument("--storage-id", default="",
                    help="rosbag2 storage_id; if empty, autodetect from metadata.yaml")
    ap.add_argument("--gt-json", type=Path, default=None,
                    help="Optional GT JSON; used with --max-dist-m to filter frames")
    ap.add_argument("--max-dist-m", type=float, default=0.0,
                    help="Only keep frames where robot is within this distance of any GT (0 disables)")
    ap.add_argument("--tf-base-frame", default="base_footprint")
    ap.add_argument("--tf-world-frame", default="map")
    args = ap.parse_args()

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Lazy imports — keep top-level cheap so --help works without ROS sourced.
    try:
        import rosbag2_py
        from cv_bridge import CvBridge
        from rclpy.serialization import deserialize_message
        from rosidl_runtime_py.utilities import get_message
        import cv2
    except ImportError as exc:
        raise SystemExit(
            f"Missing ROS Python deps ({exc}). Source: "
            "`source /opt/ros/humble/setup.bash && source install/setup.bash`."
        )

    # Detect storage id from metadata.yaml if not provided
    storage_id = args.storage_id
    if not storage_id:
        meta_path = Path(args.bag) / "metadata.yaml"
        if meta_path.exists():
            import yaml
            md = yaml.safe_load(meta_path.read_text())
            storage_id = (
                md.get("rosbag2_bagfile_information", {}).get("storage_identifier")
                or "sqlite3"
            )
        else:
            storage_id = "sqlite3"

    storage = rosbag2_py.StorageOptions(uri=args.bag, storage_id=storage_id)
    converter = rosbag2_py.ConverterOptions(input_serialization_format="cdr",
                                            output_serialization_format="cdr")
    reader = rosbag2_py.SequentialReader()
    reader.open(storage, converter)

    type_map: dict[str, str] = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if args.image_topic not in type_map:
        raise SystemExit(
            f"Image topic {args.image_topic!r} not in bag. Available: {sorted(type_map)}"
        )

    img_msg_type = get_message(type_map[args.image_topic])
    tf_msg_type = get_message(type_map[args.tf_topic]) if args.tf_topic in type_map else None
    bridge = CvBridge()

    # Build a TF lookup table keyed by (parent, child) → list of (stamp_ns, T_4x4)
    tf_chain: dict[tuple[str, str], list[tuple[int, np.ndarray]]] = {}

    def _store_tf_msg(msg, is_static: bool) -> None:
        for tr in msg.transforms:
            parent = tr.header.frame_id
            child = tr.child_frame_id
            t = tr.transform
            stamp_ns = int(tr.header.stamp.sec) * 1_000_000_000 + int(tr.header.stamp.nanosec)
            T = np.eye(4)
            T[:3, 3] = [t.translation.x, t.translation.y, t.translation.z]
            qx, qy, qz, qw = t.rotation.x, t.rotation.y, t.rotation.z, t.rotation.w
            xx, yy, zz = qx * qx, qy * qy, qz * qz
            xy, xz, yz = qx * qy, qx * qz, qy * qz
            wx, wy, wz = qw * qx, qw * qy, qw * qz
            R = np.array([
                [1 - 2 * (yy + zz), 2 * (xy - wz),     2 * (xz + wy)],
                [2 * (xy + wz),     1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy),     2 * (yz + wx),     1 - 2 * (xx + yy)],
            ])
            T[:3, :3] = R
            tf_chain.setdefault((parent, child), []).append((stamp_ns, T))

    def _lookup_chain(world: str, base: str, stamp_ns: int) -> np.ndarray | None:
        # Try to chain world → ... → base by greedy parent walk; return base xy in world if any.
        # We only need (world, base) nearest-time transform. Robust DFS over `tf_chain`.
        adj: dict[str, list[tuple[str, list[tuple[int, np.ndarray]]]]] = {}
        for (p, c), entries in tf_chain.items():
            adj.setdefault(p, []).append((c, entries))
            adj.setdefault(c, []).append(
                (p, [(s, np.linalg.inv(T)) for s, T in entries])
            )

        # BFS from world looking for base
        from collections import deque
        Q = deque([(world, np.eye(4))])
        visited = {world}
        while Q:
            cur, T_acc = Q.popleft()
            if cur == base:
                return T_acc
            for nxt, entries in adj.get(cur, []):
                if nxt in visited:
                    continue
                visited.add(nxt)
                # Pick TF entry nearest in time to stamp_ns
                _, T_step = min(entries, key=lambda e: abs(e[0] - stamp_ns))
                Q.append((nxt, T_acc @ T_step))
        return None

    gt_xy = _load_gt_xy(args.gt_json) if args.gt_json else []
    saved = 0
    frame_idx = 0
    seen_images = 0

    print(f"[dump_frames] bag={args.bag} image_topic={args.image_topic} every_n={args.every_n}")
    print(f"[dump_frames] gt_xy_count={len(gt_xy)} max_dist_m={args.max_dist_m}")

    while reader.has_next():
        topic, raw, stamp_ns = reader.read_next()
        if topic in (args.tf_topic, args.tf_static_topic) and tf_msg_type is not None:
            try:
                msg = deserialize_message(raw, tf_msg_type)
                _store_tf_msg(msg, is_static=(topic == args.tf_static_topic))
            except Exception as e:
                print(f"  tf deserialize fail: {e}")
            continue
        if topic != args.image_topic:
            continue
        seen_images += 1
        if (seen_images - 1) % args.every_n != 0:
            continue
        img_msg = deserialize_message(raw, img_msg_type)
        try:
            cv_img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="bgr8")
        except Exception as e:
            print(f"  cv_bridge fail @ {stamp_ns}: {e}")
            continue
        # Try TF lookup
        robot_xy = None
        dist_to_gt = None
        T = _lookup_chain(args.tf_world_frame, args.tf_base_frame, stamp_ns)
        if T is not None:
            robot_xy = [float(T[0, 3]), float(T[1, 3])]
            if gt_xy:
                dist_to_gt = min(math.hypot(robot_xy[0] - g[0], robot_xy[1] - g[1])
                                 for _, g in gt_xy)
        if args.max_dist_m > 0.0 and (dist_to_gt is None or dist_to_gt > args.max_dist_m):
            continue
        out_png = out_dir / f"frame_{frame_idx:04d}.png"
        out_json = out_dir / f"frame_{frame_idx:04d}.json"
        cv2.imwrite(str(out_png), cv_img)
        out_json.write_text(json.dumps({
            "timestamp_ns": stamp_ns,
            "robot_xy": robot_xy,
            "dist_to_nearest_gt_m": dist_to_gt,
            "image_topic": args.image_topic,
            "image_shape": list(cv_img.shape),
        }, indent=2))
        saved += 1
        frame_idx += 1
        if saved >= args.max_frames:
            break

    manifest = {
        "bag": args.bag,
        "image_topic": args.image_topic,
        "every_n": args.every_n,
        "max_frames": args.max_frames,
        "saved": saved,
        "seen_images": seen_images,
        "tf_base_frame": args.tf_base_frame,
        "tf_world_frame": args.tf_world_frame,
        "gt_json": str(args.gt_json) if args.gt_json else None,
        "max_dist_m": args.max_dist_m,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[dump_frames] saved={saved}/{seen_images} → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
