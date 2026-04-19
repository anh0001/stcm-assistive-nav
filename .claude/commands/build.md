---
description: Build stcm package with colcon and source workspace
---

Build stcm package + source result. Run from repo root.

Steps:
1. Run `colcon build --packages-select stcm`
2. Report any build errors from stderr
3. Remind user to `source install/setup.bash` in their shell (cannot persist from Claude session)

If build fails, diagnose by reading error output. Common causes:
- Missing pip dep (fix: `python3 -m pip install --user <pkg>` under `$PYTHONUSERBASE`)
- Shebang mismatch (nodes must use `/usr/bin/python3`)
- `PYTHONUSERBASE` not exported in Claude env
