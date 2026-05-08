from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image, ImageOps


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
COPY_EXTENSIONS = {".txt"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Resize dataset images so the longest side is at most 1536px."
    )
    parser.add_argument("--input", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("dataset_1536"))
    parser.add_argument("--max-side", type=int, default=1536)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in the output directory.",
    )
    return parser.parse_args()


def output_path_for(source: Path, input_root: Path, output_root: Path) -> Path:
    return output_root / source.relative_to(input_root)


def resize_image(source: Path, destination: Path, max_side: int, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)

    with Image.open(source) as image:
        image = ImageOps.exif_transpose(image)
        width, height = image.size
        longest_side = max(width, height)

        if longest_side > max_side:
            scale = max_side / longest_side
            new_size = (round(width * scale), round(height * scale))
            image = image.resize(new_size, Image.Resampling.LANCZOS)

        suffix = destination.suffix.lower()
        if suffix in {".jpg", ".jpeg"} and image.mode in {"RGBA", "LA", "P"}:
            image = image.convert("RGB")

        image.save(destination)

    return True


def copy_sidecar(source: Path, destination: Path, overwrite: bool) -> bool:
    if destination.exists() and not overwrite:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return True


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_root = args.output.resolve()

    if not input_root.is_dir():
        raise SystemExit(f"Input directory not found: {input_root}")
    if input_root == output_root:
        raise SystemExit("Input and output directories must be different.")

    resized = 0
    copied = 0
    skipped = 0

    for source in input_root.rglob("*"):
        if not source.is_file():
            continue

        suffix = source.suffix.lower()
        destination = output_path_for(source, input_root, output_root)

        if suffix in IMAGE_EXTENSIONS:
            if resize_image(source, destination, args.max_side, args.overwrite):
                resized += 1
            else:
                skipped += 1
        elif suffix in COPY_EXTENSIONS:
            if copy_sidecar(source, destination, args.overwrite):
                copied += 1
            else:
                skipped += 1

    print(f"Input:   {input_root}")
    print(f"Output:  {output_root}")
    print(f"Resized images: {resized}")
    print(f"Copied captions: {copied}")
    print(f"Skipped existing files: {skipped}")


if __name__ == "__main__":
    main()
