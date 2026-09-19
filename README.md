# OCTA preprocessing for further processing

# NOTE: the folder OCTA-autoseg is a copy from
# https://github.com/aiforvision/OCTA-autosegmentation


This repo is a minimal OCTA preprocessing pipeline for further vessel analysis.

## Scope

The repo primarily contains code for:
- ROI cropping
- baseline/flicker PNG pairing
- flicker to baseline registration
- post-registration trimming
- autosegmentation on the trimmed outputs

## Recommendations

- Do **not** register `CC` or `CH`
- Do **not** use binary masks before registration
- Do use binary masks before further analysis
- Threshold baseline/flicker consistently, ideally with a joint threshold per pair

## Pipeline

1. Crop OCTA scans to the ROI
2. Pair baseline (`B...`) and flicker (`F...`) PNGs
3. Register flicker to baseline using:
   1. phase cross-correlation
   2. SimpleITK affine
   3. SimpleITK B-spline
4. Trim the invalid edge band from the registered PNGs
5. Run autosegmentation on the trimmed registered PNGs

## Scripts

### 1) Crop ROI

Use `scripts/ROI_cropping.py` to crop scans and write PNG outputs.

```bash
uv run python scripts/ROI_cropping.py \
  --input_dir ./input \
  --input_ext tif \
  --output_dir ./cropped \
  --roi_size 968
```

Notes:
- The script can preserve subfolder structure
- It writes `problematic.csv` for images with suspect crops

### 2) Pair and register baseline/flicker PNGs

Use `scripts/register_paired_pngs.py` after ROI cropping.

```bash
uv run python scripts/register_paired_pngs.py \
  --input-dir ./cropped \
  --output-dir ./output/registered/baseline
```

This script:
- finds `B...` / `F...` PNG pairs automatically
- skips `CC` and `CH` by default
- exports unregistered copies to `paired_unregistered/`
- writes registered shared-overlap crops to `cropped_registered/`

Useful flags:
- `--dry-run`
- `--max-pairs 1`
- `--bspline-grid-spacing 96`
- `--crop-margin 4`

### 3) Trim registered PNG edges

Use `scripts/trim_band.py` on the registered PNG outputs.

```bash
uv run python scripts/trim_band.py \
  --input-dir ./output/registered/baseline/cropped_registered \
  --output-dir ./output/registered/baseline/cropped_registered_trimmed \
  --band-threshold 5
```

This script:
- removes the low-quality edge band introduced by registration
- writes trimmed images to `cropped_registered_trimmed/`

### 4) Run autosegmentation

Use `scripts/autosegmentation.py` on the trimmed registered PNGs.

```bash
uv run python scripts/autosegmentation.py \
  --config_file ./OCTA-autoseg/docker/trained_models/ves_seg-S-GAN/config.yml \
  --Output.save_dir ./OCTA-autoseg/docker/trained_models/ves_seg-S-GAN \
  --Test.data.image.files "./output/registered/baseline/cropped_registered_trimmed/*.png" \
  --Test.save_dir ./output/masked_autoseg/baseline \
  --epoch 30
```

This script:
- runs the vessel segmentation model on trimmed registered images
- writes masks to the chosen output directory
