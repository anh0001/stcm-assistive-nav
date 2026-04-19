---
description: Clean build artifacts and rebuild stcm from scratch
---

Nuke build artifacts + rebuild. Run from repo root.

Steps:
1. `rm -rf build/ install/ log/`
2. `colcon build --packages-select stcm`
3. Report errors if any
4. Remind user to re-source `install/setup.bash`

Use when:
- Entry points / setup.py changed
- Stale install after rename/delete of files
- Mysterious import errors not solved by source
