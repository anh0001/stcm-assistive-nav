from __future__ import annotations

import os
from pathlib import Path


def _find_models_directory(start: Path | None = None) -> Path | None:
    """Search upwards from start (or this file) to locate a repo-level models dir."""
    search_root = Path(start or __file__).resolve()
    for candidate in (search_root, *search_root.parents):
        models_dir = candidate / "models"
        if models_dir.is_dir():
            return models_dir
    return None


def get_default_checkpoint_dir() -> Path:
    """
    Resolve the default checkpoint directory used by STCM.

    Preference order:
      1. STCM_CKPT_DIR environment variable.
      2. ./models directory in the repo/workspace (if present).
      3. ./models relative to the current working directory (if present).
      4. ~/.stcm/ckpts fallback.
    """
    env_override = os.environ.get("STCM_CKPT_DIR")
    if env_override:
        return Path(env_override).expanduser()

    repo_models = _find_models_directory()
    if repo_models:
        return repo_models

    cwd_models = (Path.cwd() / "models")
    if cwd_models.is_dir():
        return cwd_models

    return Path.home() / ".stcm" / "ckpts"


__all__ = ["get_default_checkpoint_dir"]
