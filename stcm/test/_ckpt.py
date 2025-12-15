"""
Helper module to ensure unit tests use the workspace checkpoints directory.
"""

import os

from stcm.core.checkpoints import get_default_checkpoint_dir

TEST_CKPT_DIR = get_default_checkpoint_dir()
os.environ.setdefault("STCM_CKPT_DIR", str(TEST_CKPT_DIR))

__all__ = ["TEST_CKPT_DIR"]
