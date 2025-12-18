import sys
from pathlib import Path

from absl import app, logging
from PIL import Image as PILImg

# Allow running this file directly (python stcm/test/test_depth_anything.py ...)
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

from stcm.core.vision_utils import apply_matplotlib_colormap
from stcm.core.perception import DepthAnythingPredictor


def main(argv):
    # Path to the input image
    image_path = argv[1]

    try:
        logging.info("Initialize object detectors")
        depth_any = DepthAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        w, h =image_pil.size

        logging.info("Depth Anything prediction")
        depth_pil, raw_depth_output = depth_any.predict(image_pil)

        logging.info("Convert depth values to heatmap format")
        # colormap ref: https://github.com/yuki-koyama/pycolormap?tab=readme-ov-file
        depth_to_colomap_pil = apply_matplotlib_colormap(depth_pil, colormap_name='inferno')

        # Save the output image to repo/output folder
        repo_root = PKG_ROOT.parent
        output_dir = repo_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (Path(image_path).stem + "_depth.png")
        depth_to_colomap_pil.save(output_path)
        logging.info(f"Saved depth map to: {output_path}")

        # Try to display (may not work in headless environments)
        try:
            depth_to_colomap_pil.show()
        except Exception as e:
            logging.warning(f"Could not display image: {e}")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    app.run(main)
