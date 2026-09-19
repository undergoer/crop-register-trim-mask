"""
trim_band.py

Removes dark padding bands that may appear on any edge of registered flicker images
(an artefact of the registration warp and overlap cropping).

Strategy
--------
1. Scan every flicker image to find the non-dark content box using row/column
   mean thresholds.
2. Intersect those content boxes across all flicker images so the same crop is
   applied to every image (baseline and flicker alike), guaranteeing:
   - No band remains in any image.
   - Baseline / flicker pairs stay perfectly aligned.
   - All output images share identical dimensions.
3. Apply that shared crop to both baseline and flicker images.

Usage
-----
    uv run python scripts/trim_band.py \
        --input-dir  ./output/registered/baseline/cropped_registered \
        --output-dir ./output/registered/baseline/cropped_registered_trimmed

Optional flags:
    --band-threshold  INT   Row/column-mean value above which content is present (default: 5)
    --dry-run               Print detected band widths and planned crop; do not write files.
"""

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def save_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode="L").save(path)


def detect_content_bbox(img: np.ndarray, threshold: float) -> tuple[int, int, int, int]:
    """Return (row_start, row_end, col_start, col_end) of thresholded image content."""
    row_means = img.mean(axis=1)
    col_means = img.mean(axis=0)
    content_rows = np.where(row_means > threshold)[0]
    content_cols = np.where(col_means > threshold)[0]

    if len(content_rows) == 0 or len(content_cols) == 0:
        raise ValueError("Image contains no content above threshold.")

    row_start = int(content_rows[0])
    row_end = int(content_rows[-1]) + 1
    col_start = int(content_cols[0])
    col_end = int(content_cols[-1]) + 1
    return row_start, row_end, col_start, col_end


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trim dark padding bands from any edge of registered flicker/baseline PNG pairs."
    )
    parser.add_argument(
        "--input-dir", required=True,
        help="Directory containing *_baseline_registered_crop.png and *_flicker_registered_crop.png files."
    )
    parser.add_argument(
        "--output-dir", required=True,
        help="Directory to write trimmed images into."
    )
    parser.add_argument(
        "--band-threshold", type=float, default=5.0,
        help="Row/column-mean pixel value above which pixels are considered content (default: 5)."
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Detect and report band widths without writing any files."
    )
    args = parser.parse_args()

    input_dir  = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    flicker_files = sorted(input_dir.glob("*_flicker_registered_crop.png"))
    if not flicker_files:
        raise RuntimeError(f"No flicker images found in {input_dir}")

    # ---- Pass 1: detect content boxes in every flicker image ----
    content_boxes: dict[str, tuple[int, int, int, int]] = {}

    for fpath in flicker_files:
        key = fpath.name.replace("_flicker_registered_crop.png", "")
        img = load_gray(fpath)
        row_start, row_end, col_start, col_end = detect_content_bbox(img, args.band_threshold)
        content_boxes[key] = (row_start, row_end, col_start, col_end)
        top = row_start
        bottom = img.shape[0] - row_end
        left = col_start
        right = img.shape[1] - col_end
        print(
            f"  {key}: top={top}, bottom={bottom}, left={left}, right={right}, "
            f"content_rows={row_start}:{row_end}, content_cols={col_start}:{col_end}, shape={img.shape}"
        )

    crop_row_start = max(box[0] for box in content_boxes.values())
    crop_row_end = min(box[1] for box in content_boxes.values())
    crop_col_start = max(box[2] for box in content_boxes.values())
    crop_col_end = min(box[3] for box in content_boxes.values())

    if crop_row_start >= crop_row_end or crop_col_start >= crop_col_end:
        raise RuntimeError(
            "Computed crop is empty. Try lowering --band-threshold or inspect the registered images."
        )

    out_h = crop_row_end - crop_row_start
    out_w = crop_col_end - crop_col_start

    print("\nShared crop across all pairs:")
    print(f"  rows {crop_row_start}:{crop_row_end}")
    print(f"  cols {crop_col_start}:{crop_col_end}")
    print(f"  Uniform output size: {out_h} rows × {out_w} cols")

    if args.dry_run:
        print("\n--dry-run: no files written.")
        return

    # ---- Pass 2: crop and save both baseline and flicker for every key ----
    for key in sorted(content_boxes.keys()):
        for role in ("baseline", "flicker"):
            src = input_dir / f"{key}_{role}_registered_crop.png"
            dst = output_dir / f"{key}_{role}_registered_crop.png"
            if not src.exists():
                print(f"  WARNING: {src} not found, skipping.")
                continue
            img = load_gray(src)
            cropped = img[crop_row_start:crop_row_end, crop_col_start:crop_col_end]
            save_png(dst, cropped)
            print(f"  Saved {dst.name}  {img.shape} → {cropped.shape}")

    print(f"\nDone. Trimmed images written to: {output_dir}")


if __name__ == "__main__":
    main()

