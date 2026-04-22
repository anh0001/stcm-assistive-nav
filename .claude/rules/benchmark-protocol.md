# Benchmark Protocol Rules (JACIII Jc26-0002 Revision)

Runtime baseline contract for Table D (AE-4 / R1-4). Authoritative for `benchmark` skill runs.

## Hardware disclosure (mandatory)

Record once per machine, cite in Table D caption:
- CPU model + core count
- GPU model + VRAM
- RAM size
- OS + ROS 2 distro (Humble)
- CUDA toolkit + driver
- PyTorch version

## Modules to instrument

| Module | Measurement hook | Target cadence |
|--------|------------------|----------------|
| FAST-LIO2 | `/odom` or `/Odometry` publish | 10–20 Hz |
| place-GNG | per-pose update fn in `map_utils` | ≤ pose rate |
| instance-GNG | per-detection update fn | ≤ detection rate |
| GroundingDINO + SAM | `predict()` call site in `core/perception.py` | ≥ 1 Hz |
| LiDAR–image assoc | mask ∩ projected (u,v) loop | per detection |
| temporal fusion | node-score update in `nodes/semantic_map_builder.py` | per detection |
| LLM grounding | LangChain call in grounding interface | per command |
| Nav2 planning/control | `bt_navigator` action + `controller_server` cycle | Nav2 rates |

## Metrics per module

- mean + p50 + p95 latency (ms), ≥100 iterations warm
- effective rate (Hz / FPS)
- CPU % (psutil per-proc)
- GPU util % + VRAM MB (nvidia-smi dmon)
- RAM RSS MB

No median-of-5 theatre. Publish full sample if < 100 iters.

## Navigation-critical vs asynchronous split

Critical (must hit real-time in closed loop):
- FAST-LIO2, Nav2 planner + controller

Asynchronous (allowed < 1 Hz, run on own thread):
- GroundingDINO + SAM, LLM grounding

Table D caption must state: "Asynchronous modules do not block navigation loop; latency measured independently."

## Silent-failure traps

Use `silent-failure-hunter` agent before publishing numbers. Common fakes:
- early-return on missing TF → fake low p95
- cached result on repeated call → fake Hz
- GPU warmup included → bogus mean
- detection loop skipping frames → fake FPS

## Output

`results/bench/runtime_<sha>.json` schema:
```json
{
  "sha": "...",
  "hw": {...},
  "modules": {
    "fast_lio2": {"mean_ms": 42.1, "p50_ms": 40.0, "p95_ms": 68.3, "hz": 20.0, "cpu_pct": 35.2, "gpu_pct": 0.0, "ram_mb": 340, "n": 600},
    ...
  }
}
```

LaTeX Table D generated via `scripts/render_table_d.py`.
