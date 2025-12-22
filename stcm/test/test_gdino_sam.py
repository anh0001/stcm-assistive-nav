import sys
from pathlib import Path

from absl import app, logging
from PIL import Image as PILImg

# Allow running this file directly (python stcm/test/test_gdino_sam.py ...)
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

from stcm.core.vision_utils import annotate, overlay_masks
from stcm.core.perception import GroundingDINOObjectPredictor, SegmentAnythingPredictor


def main(argv):
    # Path to the input image
    # argv[0] is the script name, argv[1] is the actual first argument
    if len(argv) < 2:
        print("Usage: python test_gdino_sam.py <image_path>")
        return
    image_path = argv[1]
    text_prompt =  'objects . dark thermos bottle .'

    try:
        logging.info("Initialize object detectors")
        gdino = GroundingDINOObjectPredictor()
        SAM = SegmentAnythingPredictor()

        logging.info("Open the image and convert to RGB format")
        image_pil = PILImg.open(image_path).convert("RGB")
        
        logging.info("GDINO: Predict bounding boxes, phrases, and confidence scores")
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)

        logging.info("GDINO post processing")
        w, h = image_pil.size # Get image width and height 
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)

        logging.info("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)

        logging.info("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), image_pil_bboxes, gdino_conf, phrases)

        # Save the output image to repo/output folder
        repo_root = PKG_ROOT.parent
        output_dir = repo_root / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (Path(image_path).stem + "_annotated.png")
        bbox_annotated_pil.save(output_path)
        logging.info(f"Saved annotated image to: {output_path}")

        # Try to display (may not work in headless environments)
        try:
            bbox_annotated_pil.show()
        except Exception as e:
            logging.warning(f"Could not display image: {e}")

    except Exception as e:
        # Handle unexpected errors
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Run the main function with the input image path
    app.run(main)
