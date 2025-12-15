import sys
from pathlib import Path

import numpy as np
from absl import app, logging
from PIL import Image as PILImg

# Allow running this file directly (python stcm/test/test_sam.py ...)
TEST_DIR = Path(__file__).resolve().parent
PKG_ROOT = TEST_DIR.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

try:
    from stcm.test._ckpt import TEST_CKPT_DIR  # noqa: F401
except ModuleNotFoundError:
    if str(TEST_DIR) not in sys.path:
        sys.path.insert(0, str(TEST_DIR))
    from _ckpt import TEST_CKPT_DIR  # noqa: F401

from stcm.core.vision_utils import overlay_masks
from stcm.core.perception import SegmentAnythingPredictor


def main(argv):
    # Path to the input image
    image_path = argv[0]

    try:
        logging.info("Initialize object detectors")
        SAM = SegmentAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        w, h =image_pil.size

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, prompt_bboxes=np.array([0,0,w,h]))
        # if prompt_bboxes is None, SAM will generate masks for the entire image: todo not yet complete
        # image_pil_bboxes, masks = SAM.predict(image_pil, prompt_bboxes=None)

        overlay_masks(image_pil,masks).show()


    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    # Run the main function with the input image path
    # app.run(main, ['imgs/color-000078.png'])
    # app.run(main, ['imgs/color-000019.png'])
    app.run(main, ['imgs/irvl-clutter-test.png'])
