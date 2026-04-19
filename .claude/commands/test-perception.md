---
description: Run standalone perception tests (GDINO+SAM, depth) outside ROS
argument-hint: [image_path]
---

Run perception unit tests w/o ROS. Uses `stcm/test/` scripts + shared checkpoint helper.

Default image: `stcm/imgs/irvl-clutter-test.png`. Override via `$ARGUMENTS`.

Run:
1. `python3 stcm/test/test_gdino_sam.py <image_path>` — GDINO detect + MobileSAM seg
2. `python3 stcm/test/test_depth_anything.py stcm/imgs/color-000089.png` — depth est

Pre-checks:
- `PYTHONUSERBASE` exported
- Checkpoints downloaded
- Run from repo root (scripts use relative paths)

Output = annotated PNGs in test dir. Report pass/fail + paths.
