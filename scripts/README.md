# MAVREC vehicle crop subset generator

This repository **already ships** a ready-to-run demo under [`datasets/mavrec_car_van/`](../datasets/mavrec_car_van/) (CC BY 4.0). Use this script only if you need to **regenerate** that subset after accepting the Hugging Face gated terms and downloading MAVREC from Drive.

Build a **car vs van** binary classification folder from the MAVREC detection dataset (official **val** split). Output is G-CBM-compatible: cropped images + `image_path,labels` CSVs.

**Licence / attribution:** MAVREC is **CC BY 4.0** ([deed](https://creativecommons.org/licenses/by/4.0/)). Credit the creators (Dutta, Das, Nielsen, Chakraborty, Shah; CVPR 2024) when regenerating or redistributing crops. Canonical text + BibTeX: [NOTICE](../NOTICE); also [mavrec.github.io](https://mavrec.github.io/), [Hugging Face](https://huggingface.co/datasets/rjccv/MAVREC).

## Quick start

1. Accept the gated license and follow Drive download steps (below).
2. Point the script at the folder that contains `MAVREC/` (input).
3. By default the subset is written under `datasets/` in this repo (overwrites / refreshes `mavrec_car_van/` if present).

```bash
# from the repo root (parent of scripts/)
python scripts/generate_mavrec_vehicle_crops.py \
  --mavrec-parent /path/to/folder_that_contains_MAVREC
```

Optional: write somewhere else with `--output-root /path/to/out`.

`--dataset-root` is an alias for `--mavrec-parent`.

## Inputs vs outputs

| Role | Path | What it is |
| ---- | ---- | ---------- |
| **Input** | `<mavrec-parent>/MAVREC/` | Raw MAVREC download (val images + annotations) |
| **Output** | `<output-root>/mavrec_car_van/` | car (0) vs van (1) |

Defaults:

- `--mavrec-parent` → parent of `scripts/` (repo root)
- `--output-root` → `<repo>/datasets` (created automatically)

After a successful run you should see:

```
datasets/
  mavrec_car_van/
```

The subset has `images/`, `train.csv`, `validation.csv`, `test.csv`, `nmf.csv`, `all.csv`, `class_names.txt`, and a short README with split counts.

`MAVREC/` itself is **not** created by this script — you download that separately.

## Prerequisites

### 1. Hugging Face access (gated)

Open [https://huggingface.co/datasets/rjccv/MAVREC](https://huggingface.co/datasets/rjccv/MAVREC), accept the license, then:

```bash
huggingface-cli login
```

### 2. Download the data

The Hugging Face page is mostly metadata. Full labelled data is linked from `ACCESS_INSTRUCTIONS.md` inside the Hub clone (Google Drive).

You need at least:

- `supervised_annotations.zip`
- `val.zip` (unzip into `val/`)

### 3. Expected MAVREC layout

```
<mavrec-parent>/
  MAVREC/
    ACCESS_INSTRUCTIONS.md
    data/
      labelled/
        supervised_annotations.zip
        val/
          aerial/*.PNG
          ground/*.PNG
```

If you only have `val.zip`:

```bash
cd MAVREC/data/labelled
unzip -q val.zip -d .
# expect val/aerial and val/ground
```

### 4. Python

```bash
pip install Pillow
```

## Run examples

```bash
python scripts/generate_mavrec_vehicle_crops.py \
  --mavrec-parent /path/to/folder_with_MAVREC

# custom output location
python scripts/generate_mavrec_vehicle_crops.py \
  --mavrec-parent /path/to/folder_with_MAVREC \
  --output-root /path/to/my_datasets
```

If `MAVREC/` already sits next to `scripts/` (same parent), you can omit `--mavrec-parent`.

## Processing decisions (fixed in the script)

| Setting       | Value                                                    |
| ------------- | -------------------------------------------------------- |
| Source split  | Official MAVREC **val** only (`train.zip` not required)  |
| Views         | Both `aerial` and `ground` frames                        |
| Min box size  | width and height ≥ 32 px                                 |
| Bbox padding  | 5%                                                       |
| Resize        | max side 256, JPEG quality 90                            |
| Cap per class | `min(800, available_car, available_van)` after shuffle   |
| Seed          | 42                                                       |
| Labels        | car → **0**, van → **1**                                 |
| Splits        | per-class stratified ~70% / 15% / 15% train / val / test |
| `nmf.csv`     | copy of `train.csv`                                      |

Approximate size on val (≥32 px boxes): **~552 images per class**.

## CSV format

```csv
image_path,labels
car/scene_1_..._a123.jpg,0
van/scene_2_..._a456.jpg,1
```

Paths are relative to `mavrec_car_van/images/`.

## Notes

- This is a **pilot** built from val only. For paper-quality splits, re-run (or extend the script) once `train.zip` is available.
