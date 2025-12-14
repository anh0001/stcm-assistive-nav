import argparse
import logging
from pathlib import Path

import requests
from tqdm import tqdm


DEFAULT_MODELS = {
    "groundingdino": {
        "url": "https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth",
        "filename": "groundingdino_swint_ogc.pth",
        "subdir": "gdino",
        "description": "GroundingDINO SwinT checkpoint used for zero-shot text-to-box detection.",
    },
    "mobilesam": {
        "url": "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt",
        "filename": "vit_t.pth",
        "subdir": "mobilesam",
        "description": "MobileSAM weights for mask extraction.",
    },
    "depth-anything": {
        "url": "https://huggingface.co/spaces/camenduru/Depth-Anything/resolve/main/depth_anything_vitb14.pth",
        "filename": "depth_anything_vitb14.pth",
        "subdir": "depth_anything",
        "description": "Depth-Anything ViT-B checkpoint for monocular depth estimation.",
    },
}


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        logging.info("Skipping %s because it already exists.", destination)
        return

    logging.info("Downloading %s to %s", url, destination)
    with requests.get(url, stream=True, timeout=30) as response:
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                handle.write(chunk)
                progress_bar.update(len(chunk))
        progress_bar.close()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download model checkpoints used by the STCM semantic mapping pipeline."
    )
    parser.add_argument(
        "--target",
        type=Path,
        default=Path.home() / ".stcm" / "ckpts",
        help="Directory where checkpoints will be stored (default: ~/.stcm/ckpts).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(DEFAULT_MODELS.keys()),
        default=list(DEFAULT_MODELS.keys()),
        help="Subset of models to download. Defaults to all available models.",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Print the available models and exit.",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    if args.list:
        for name, data in DEFAULT_MODELS.items():
            print(f"{name}\n  url: {data['url']}\n  target: {data['subdir']}/{data['filename']}\n  {data['description']}\n")
        return

    for model_name in args.models:
        model_data = DEFAULT_MODELS[model_name]
        destination = args.target / model_data["subdir"] / model_data["filename"]
        download_file(model_data["url"], destination)

    logging.info("All requested checkpoints are ready in %s", args.target.resolve())


if __name__ == "__main__":
    main()
