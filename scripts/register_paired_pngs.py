import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import shift as ndi_shift
from skimage import exposure, filters
from skimage.registration import phase_cross_correlation


def load_gray_png(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"), dtype=np.float32)


def save_uint8_png(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(image, 0, 255).astype(np.uint8)
    Image.fromarray(clipped, mode="L").save(path)


def normalize01(image: np.ndarray) -> np.ndarray:
    lo = float(image.min())
    hi = float(image.max())
    return (image - lo) / (hi - lo + 1e-6)


def preprocess_for_registration(image: np.ndarray) -> np.ndarray:
    """Prepare an image for registration via CLAHE + Frangi vesselness blending.

    NOTE on binary masks before registration:
        Applying a binary vessel mask *before* registration is generally a bad idea.
        Mattes Mutual Information and Mean Squares both rely on continuous intensity
        gradients to drive the optimiser. Binarising removes those gradients, leaving
        only 0/1 edge transitions — too sparse for stable convergence, especially for
        the deformable B-spline stage. What *is* useful is a valid-overlap binary mask
        applied *after* registration to crop the common field-of-view, which is exactly
        what ``run_registration`` does with ``common_mask``.
    """
    norm = normalize01(image)
    equalized = exposure.equalize_adapthist(norm, clip_limit=0.015)
    vesselness = filters.frangi(equalized, sigmas=np.arange(1, 4), black_ridges=False)
    vesselness = normalize01(vesselness)
    # Blend vesselness and intensity contrast for more stable multimodal matching.
    return (0.6 * equalized + 0.4 * vesselness).astype(np.float32)


def pair_key_from_filename(path: Path) -> tuple[str | None, str | None]:
    name = path.name
    if name.lower().endswith(".tif.png"):
        stem = name[:-8]
    else:
        stem = path.stem

    match = re.match(r"^([BbFf])(.*)$", stem)
    if not match:
        return None, None

    prefix = match.group(1).upper()
    key = match.group(2)
    return prefix, key


def modality_from_key(key: str) -> str:
    """Extract modality suffix from a pair key, e.g. '15-SVP' -> 'SVP', '1-CC' -> 'CC'."""
    parts = key.rsplit("-", 1)
    return parts[-1].upper() if len(parts) > 1 else key.upper()


def collect_pairs(
    input_dir: Path,
    pattern: str,
    skip_modalities: set[str] | None = None,
) -> tuple[list[tuple[str, Path, Path]], list[Path]]:
    candidates = sorted(input_dir.rglob(pattern))
    ignored = []
    grouped: dict[str, dict[str, Path]] = {}
    skip_modalities = {m.upper() for m in skip_modalities} if skip_modalities else set()

    for file_path in candidates:
        prefix, key = pair_key_from_filename(file_path)
        if prefix is None or key is None:
            ignored.append(file_path)
            continue
        modality = modality_from_key(key)
        if modality in skip_modalities:
            ignored.append(file_path)
            continue
        grouped.setdefault(key, {})[prefix] = file_path

    pairs = []
    for key in sorted(grouped.keys()):
        group = grouped[key]
        if "B" in group and "F" in group:
            pairs.append((key, group["B"], group["F"]))

    return pairs, ignored


def bbox_from_mask(mask: np.ndarray, margin: int) -> tuple[int, int, int, int]:
    coords = np.where(mask)
    if coords[0].size == 0:
        h, w = mask.shape
        return 0, h, 0, w

    r0 = int(coords[0].min())
    r1 = int(coords[0].max()) + 1
    c0 = int(coords[1].min())
    c1 = int(coords[1].max()) + 1

    r0 = max(0, r0 + margin)
    c0 = max(0, c0 + margin)
    r1 = max(r0 + 1, r1 - margin)
    c1 = max(c0 + 1, c1 - margin)

    return r0, r1, c0, c1


def run_registration(
    fixed_img: np.ndarray,
    moving_img: np.ndarray,
    bspline_grid_spacing: int,
    crop_margin: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    fixed_reg_np = preprocess_for_registration(fixed_img)
    moving_reg_np = preprocess_for_registration(moving_img)

    shift_rc, _, _ = phase_cross_correlation(fixed_reg_np, moving_reg_np, upsample_factor=20)
    moving_img_shifted = ndi_shift(moving_img, shift=shift_rc, order=1, mode="constant", cval=0.0, prefilter=False)
    moving_reg_shifted = ndi_shift(moving_reg_np, shift=shift_rc, order=1, mode="constant", cval=0.0, prefilter=False)

    fixed_reg = sitk.GetImageFromArray(fixed_reg_np)
    moving_reg = sitk.GetImageFromArray(moving_reg_shifted)
    fixed_actual = sitk.GetImageFromArray(fixed_img.astype(np.float32))
    moving_actual = sitk.GetImageFromArray(moving_img_shifted.astype(np.float32))
    moving_valid_mask = sitk.GetImageFromArray(np.ones_like(moving_img_shifted, dtype=np.float32))

    affine_init = sitk.CenteredTransformInitializer(
        fixed_reg,
        moving_reg,
        sitk.AffineTransform(2),
        sitk.CenteredTransformInitializerFilter.GEOMETRY,
    )

    affine_reg = sitk.ImageRegistrationMethod()
    affine_reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
    affine_reg.SetMetricSamplingStrategy(affine_reg.RANDOM)
    affine_reg.SetMetricSamplingPercentage(0.2)
    affine_reg.SetInterpolator(sitk.sitkLinear)
    affine_reg.SetOptimizerAsRegularStepGradientDescent(
        learningRate=1.0,
        minStep=1e-4,
        numberOfIterations=300,
        relaxationFactor=0.5,
    )
    affine_reg.SetOptimizerScalesFromPhysicalShift()
    affine_reg.SetShrinkFactorsPerLevel([4, 2, 1])
    affine_reg.SetSmoothingSigmasPerLevel([2, 1, 0])
    affine_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    affine_reg.SetInitialTransform(affine_init, inPlace=False)

    affine_tx = affine_reg.Execute(fixed_reg, moving_reg)

    moving_reg_affine = sitk.Resample(moving_reg, fixed_reg, affine_tx, sitk.sitkLinear, 0.0, moving_reg.GetPixelID())
    moving_actual_affine = sitk.Resample(
        moving_actual, fixed_actual, affine_tx, sitk.sitkLinear, 0.0, moving_actual.GetPixelID()
    )
    moving_mask_affine = sitk.Resample(
        moving_valid_mask, fixed_actual, affine_tx, sitk.sitkNearestNeighbor, 0.0, moving_valid_mask.GetPixelID()
    )

    fixed_size = fixed_reg.GetSize()
    mesh_size = [max(2, int(round(fixed_size[0] / bspline_grid_spacing))), max(2, int(round(fixed_size[1] / bspline_grid_spacing)))]
    bspline_init = sitk.BSplineTransformInitializer(fixed_reg, mesh_size)

    bspline_reg = sitk.ImageRegistrationMethod()
    bspline_reg.SetMetricAsMeanSquares()
    bspline_reg.SetInterpolator(sitk.sitkLinear)
    bspline_reg.SetOptimizerAsLBFGSB(
        gradientConvergenceTolerance=1e-5,
        numberOfIterations=120,
        maximumNumberOfCorrections=5,
        maximumNumberOfFunctionEvaluations=1000,
        costFunctionConvergenceFactor=1e7,
    )
    bspline_reg.SetShrinkFactorsPerLevel([2, 1])
    bspline_reg.SetSmoothingSigmasPerLevel([1, 0])
    bspline_reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
    bspline_reg.SetInitialTransform(bspline_init, inPlace=False)

    bspline_tx = bspline_reg.Execute(fixed_reg, moving_reg_affine)

    moving_registered = sitk.Resample(
        moving_actual_affine,
        fixed_actual,
        bspline_tx,
        sitk.sitkLinear,
        0.0,
        moving_actual_affine.GetPixelID(),
    )
    moving_registered_mask = sitk.Resample(
        moving_mask_affine,
        fixed_actual,
        bspline_tx,
        sitk.sitkNearestNeighbor,
        0.0,
        moving_mask_affine.GetPixelID(),
    )

    fixed_np = sitk.GetArrayFromImage(fixed_actual)
    moving_np = sitk.GetArrayFromImage(moving_registered)
    common_mask = sitk.GetArrayFromImage(moving_registered_mask) > 0.5

    r0, r1, c0, c1 = bbox_from_mask(common_mask, margin=crop_margin)
    fixed_crop = fixed_np[r0:r1, c0:c1]
    moving_crop = moving_np[r0:r1, c0:c1]

    metadata = {
        "phase_shift_row_col": [float(shift_rc[0]), float(shift_rc[1])],
        "crop_bbox_r0_r1_c0_c1": [r0, r1, c0, c1],
        "affine_metric": float(affine_reg.GetMetricValue()),
        "bspline_metric": float(bspline_reg.GetMetricValue()),
        "bspline_mesh_size": mesh_size,
    }

    return fixed_crop, moving_crop, metadata


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Pair cropped OCTA PNG files (B* with F*) and register each pair using "
            "phase-correlation + SimpleITK affine + B-spline."
        )
    )
    parser.add_argument("--input-dir", required=True, help="Directory containing cropped PNG files.")
    parser.add_argument("--output-dir", required=True, help="Output directory for paired and registered images.")
    parser.add_argument("--glob-pattern", default="*.png", help="Glob pattern for PNG discovery (default: *.png).")
    parser.add_argument("--max-pairs", type=int, default=0, help="Optional cap for debug runs (0 = all pairs).")
    parser.add_argument("--bspline-grid-spacing", type=int, default=96, help="Approximate B-spline control point spacing in pixels.")
    parser.add_argument("--crop-margin", type=int, default=4, help="Pixels trimmed from valid-overlap border.")
    parser.add_argument("--dry-run", action="store_true", help="Discover/write pair list only; skip registration.")
    parser.add_argument(
        "--skip-modalities",
        default="CC,CH",
        help=(
            "Comma-separated list of modality suffixes to exclude from registration "
            "(default: 'CC,CH'). These are still cropped by ROI_cropping.py but will "
            "not be paired or registered here because CC/CH are colour/composite channels "
            "whose structure differs from single-layer vessel maps."
        ),
    )
    args = parser.parse_args()

    skip_modalities = {m.strip().upper() for m in args.skip_modalities.split(",") if m.strip()}

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    paired_dir = output_dir / "paired_unregistered"
    registered_dir = output_dir / "cropped_registered"
    logs_dir = output_dir / "logs"

    pairs, ignored = collect_pairs(input_dir, args.glob_pattern, skip_modalities=skip_modalities)
    if skip_modalities:
        print(f"Skipping modalities: {', '.join(sorted(skip_modalities))}")
    if args.max_pairs > 0:
        pairs = pairs[: args.max_pairs]

    if not pairs:
        raise RuntimeError(f"No B/F pairs found in {input_dir}")

    paired_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    pair_rows = []
    for key, baseline_path, flicker_path in pairs:
        pair_rows.append({
            "key": key,
            "baseline_png": str(baseline_path),
            "flicker_png": str(flicker_path),
        })

    with (logs_dir / "pairs.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "baseline_png", "flicker_png"])
        writer.writeheader()
        writer.writerows(pair_rows)

    for _, baseline_path, flicker_path in pairs:
        fixed = load_gray_png(baseline_path)
        moving = load_gray_png(flicker_path)
        key = pair_key_from_filename(baseline_path)[1]
        save_uint8_png(paired_dir / f"{key}_baseline.png", fixed)
        save_uint8_png(paired_dir / f"{key}_flicker_unregistered.png", moving)

    if args.dry_run:
        print(f"Discovered {len(pairs)} pairs; wrote pair list and unregistered copies to {output_dir}")
        return

    registered_dir.mkdir(parents=True, exist_ok=True)
    metrics = []

    for key, baseline_path, flicker_path in pairs:
        print(f"Registering pair: {key}")
        fixed = load_gray_png(baseline_path)
        moving = load_gray_png(flicker_path)

        fixed_crop, moving_crop, metadata = run_registration(
            fixed,
            moving,
            bspline_grid_spacing=args.bspline_grid_spacing,
            crop_margin=args.crop_margin,
        )

        save_uint8_png(registered_dir / f"{key}_baseline_registered_crop.png", fixed_crop)
        save_uint8_png(registered_dir / f"{key}_flicker_registered_crop.png", moving_crop)

        with (registered_dir / f"{key}_registration.json").open("w") as f:
            json.dump(
                {
                    "key": key,
                    "baseline_png": str(baseline_path),
                    "flicker_png": str(flicker_path),
                    **metadata,
                },
                f,
                indent=2,
            )

        metrics.append({
            "key": key,
            "affine_metric": metadata["affine_metric"],
            "bspline_metric": metadata["bspline_metric"],
            "crop_bbox": "|".join(map(str, metadata["crop_bbox_r0_r1_c0_c1"])),
        })

    with (logs_dir / "registration_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["key", "affine_metric", "bspline_metric", "crop_bbox"])
        writer.writeheader()
        writer.writerows(metrics)

    if ignored:
        with (logs_dir / "ignored_files.txt").open("w") as f:
            for path in ignored:
                f.write(str(path) + "\n")

    print(f"Done. Registered {len(pairs)} pairs. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main()
