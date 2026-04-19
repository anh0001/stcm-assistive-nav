---
description: Download model checkpoints (GroundingDINO, MobileSAM, Depth-Anything)
argument-hint: [--models name1,name2] [--target path]
---

Download ML checkpoints to `./models` (or `$STCM_CKPT_DIR`).

Default:
```bash
ros2 run stcm stcm_download_checkpoints
```

Pass through user args if provided in `$ARGUMENTS` (e.g. `--models mobilesam --target /data/ckpts`).

After download, verify expected layout:
- `models/gdino/groundingdino_swint_ogc.pth`
- `models/mobilesam/vit_t.pth`
- `models/depth_anything/depth_anything_vitb14.pth` (optional)

List available via `ros2 run stcm stcm_download_checkpoints --list`.
