#!/usr/bin/env python3
"""
Build MAVREC car vs van crop subset for G-CBM-style classification.

Output (official val split only — train.zip not required):
  <output-root>/mavrec_car_van/

Contains:
  images/{car,van}/*.jpg
  train.csv / validation.csv / test.csv / nmf.csv / all.csv
  class_names.txt
  README.md

Example:
  python scripts/generate_mavrec_vehicle_crops.py \\
    --mavrec-parent /path/to/folder_that_contains_MAVREC
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from typing import Dict, List, Tuple

from PIL import Image

SEED = 42
CROP_MAX_SIDE = 256
CROP_MIN_SIDE = 32
CROP_PAD_FRAC = 0.05
CROP_MAX_PER_CLASS = 800
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
CLASS_A = "car"  # label 0
CLASS_B = "van"  # label 1
OUTPUT_FOLDER = "mavrec_car_van"


def _resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    m = max(w, h)
    if m <= max_side:
        return img
    scale = max_side / float(m)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    return img.resize((nw, nh), Image.Resampling.BILINEAR)


def _write_csvs(out_dir: str, rows: List[Tuple[str, int, str]]) -> Dict[str, int]:
    by_split: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for path, label, split in rows:
        by_split[split].append((path, label))

    def write_simple(name: str, data: List[Tuple[str, int]]) -> None:
        with open(os.path.join(out_dir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["image_path", "labels"])
            for p, lab in data:
                w.writerow([p, lab])

    train = by_split["train"][:]
    random.shuffle(train)
    val = by_split["validation"]
    test = by_split["test"]
    write_simple("train.csv", train)
    write_simple("validation.csv", val)
    write_simple("test.csv", test)
    write_simple("nmf.csv", train[:])

    with open(os.path.join(out_dir, "all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image_path", "labels", "split"])
        for path, label, split in rows:
            w.writerow([path, label, split])

    return {
        "train": len(train),
        "validation": len(val),
        "test": len(test),
        "nmf": len(train),
        "all": len(rows),
    }


def _class_hist(rows: List[Tuple[str, int]], names: List[str]) -> str:
    c: Dict[int, int] = defaultdict(int)
    for _, lab in rows:
        c[lab] += 1
    return ", ".join(f"{names[i]}={c[i]}" for i in sorted(c))


def _load_val_coco(ann_zip: str) -> Tuple[dict, dict]:
    with zipfile.ZipFile(ann_zip) as zf:
        aerial = json.loads(zf.read("supervised_annotations/aerial/aerial_valid.json"))
        ground = json.loads(zf.read("supervised_annotations/ground/ground_val.json"))
    return aerial, ground


def _index_images(coco: dict) -> Dict[int, dict]:
    return {im["id"]: im for im in coco["images"]}


def _category_name_to_id(coco: dict) -> Dict[str, int]:
    return {c["name"]: c["id"] for c in coco["categories"]}


def _hist_split(rows: List[Tuple[str, int, str]], split: str) -> Tuple[int, int, int]:
    subset = [(p, l) for p, l, s in rows if s == split]
    c: Dict[int, int] = defaultdict(int)
    for _, l in subset:
        c[l] += 1
    return len(subset), c[0], c[1]


def write_readme(out_dir: str, stats: dict) -> None:
    a_name = CLASS_A.capitalize()
    b_name = CLASS_B.capitalize()
    rows = stats["rows"]
    tr_n, tr_a, tr_b = _hist_split(rows, "train")
    va_n, va_a, va_b = _hist_split(rows, "validation")
    te_n, te_a, te_b = _hist_split(rows, "test")
    cap = stats["cap"]

    text = f"""# MAVREC crop subset ({a_name} vs {b_name})

Binary classification subset for G-CBM-style pipelines, built by cropping COCO bounding boxes from the MAVREC val split.

## Source dataset

- **Hugging Face:** https://huggingface.co/datasets/rjccv/MAVREC
- **Google Drive (full data):** see `MAVREC/ACCESS_INSTRUCTIONS.md` after HF access is approved
- **Local source:**
  - Images: `MAVREC/data/labelled/val/{{aerial,ground}}/`
  - Annotations: `MAVREC/data/labelled/supervised_annotations.zip` (`aerial_valid.json`, `ground_val.json`)
- **Classes:** {a_name} (COCO `{CLASS_A}` → label **0**), {b_name} (COCO `{CLASS_B}` → label **1**)
- **Filter / processing:**
  - Pilot subset uses **official val only** (`train.zip` not required)
  - Full frames are multi-object; samples are **pre-cropped boxes** (not full images)
  - Skip boxes with width or height < **{CROP_MIN_SIDE}** px; pad bbox by **{int(CROP_PAD_FRAC * 100)}%**
  - Crops resized to max side **{CROP_MAX_SIDE}**, saved as JPEG under `images/{CLASS_A}/` and `images/{CLASS_B}/`
  - Cap at **{cap}** crops per class after shuffle (seed {SEED}); raw before cap: {CLASS_A}={stats['raw_a']}, {CLASS_B}={stats['raw_b']}
- **Images:** `images/{CLASS_A}/` ({stats['n_a']}), `images/{CLASS_B}/` ({stats['n_b']}); total **{stats['total']}**

## CSVs 

| File | Count | Per class |
|------|------:|-----------|
| `train.csv` / `nmf.csv` | {tr_n} | {a_name}={tr_a}, {b_name}={tr_b} |
| `validation.csv` | {va_n} | {a_name}={va_a}, {b_name}={va_b} |
| `test.csv` | {te_n} | {a_name}={te_a}, {b_name}={te_b} |
| `all.csv` | {stats['total']} | with `split` column |

**Split rule:** per-class stratified random split (~{int(TRAIN_RATIO * 100)}/{int(VAL_RATIO * 100)}/{int((1 - TRAIN_RATIO - VAL_RATIO) * 100)} train/val/test, seed {SEED}) so each split stays class-balanced on this val-only pilot.  
`nmf.csv` is the same set as `train.csv`. All CSV paths resolve to real files under `images/`.

## Notes

- This is a **pilot** subset from val only. Rebuild from train+val when `train.zip` is available for a full experiment.
- Regenerator: `scripts/generate_mavrec_vehicle_crops.py --mavrec-parent <path>`
"""
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write(text)


def build_car_van(mavrec_parent: str, output_root: str) -> dict:
    random.seed(SEED)

    out_dir = os.path.join(output_root, OUTPUT_FOLDER)
    val_img_root = os.path.join(mavrec_parent, "MAVREC", "data", "labelled", "val")
    ann_zip = os.path.join(
        mavrec_parent, "MAVREC", "data", "labelled", "supervised_annotations.zip"
    )

    if not os.path.isdir(val_img_root):
        raise FileNotFoundError(
            f"Val images not found: {val_img_root}\n"
            "Unzip MAVREC labelled/val.zip into MAVREC/data/labelled/val/ first."
        )
    if not os.path.isfile(ann_zip):
        raise FileNotFoundError(f"Annotations zip not found: {ann_zip}")

    print(f"\n=== Building {OUTPUT_FOLDER} (car vs van) ===")
    images_out = os.path.join(out_dir, "images")
    if os.path.isdir(images_out):
        shutil.rmtree(images_out)
    for name in (CLASS_A, CLASS_B):
        os.makedirs(os.path.join(images_out, name), exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    aerial_coco, ground_coco = _load_val_coco(ann_zip)
    crop_records: List[Tuple[str, int]] = []  # rel_path, label

    for view, coco in (("aerial", aerial_coco), ("ground", ground_coco)):
        name_to_cid = _category_name_to_id(coco)
        if CLASS_A not in name_to_cid or CLASS_B not in name_to_cid:
            raise KeyError(f"Missing category in COCO JSON for view={view}")
        want = {name_to_cid[CLASS_A]: 0, name_to_cid[CLASS_B]: 1}
        id_to_im = _index_images(coco)
        anns_by_img: Dict[int, list] = defaultdict(list)
        for ann in coco["annotations"]:
            if ann["category_id"] in want:
                anns_by_img[ann["image_id"]].append(ann)

        for image_id, anns in anns_by_img.items():
            meta = id_to_im[image_id]
            fname = meta["file_name"]
            src = os.path.join(val_img_root, view, fname)
            if not os.path.isfile(src):
                continue
            with Image.open(src) as im:
                im = im.convert("RGB")
                W, H = im.size
                for ai, ann in enumerate(anns):
                    x, y, w, h = ann["bbox"]
                    if w < CROP_MIN_SIDE or h < CROP_MIN_SIDE:
                        continue
                    pad_x = w * CROP_PAD_FRAC
                    pad_y = h * CROP_PAD_FRAC
                    x0 = max(0, int(x - pad_x))
                    y0 = max(0, int(y - pad_y))
                    x1 = min(W, int(x + w + pad_x))
                    y1 = min(H, int(y + h + pad_y))
                    if x1 - x0 < CROP_MIN_SIDE or y1 - y0 < CROP_MIN_SIDE:
                        continue
                    crop = im.crop((x0, y0, x1, y1))
                    crop = _resize_max_side(crop, CROP_MAX_SIDE)
                    label = want[ann["category_id"]]
                    cls_name = CLASS_A if label == 0 else CLASS_B
                    stem = os.path.splitext(fname)[0]
                    out_name = f"{stem}_b{ai}_a{ann['id']}.jpg"
                    out_rel = f"{cls_name}/{out_name}"
                    crop.save(os.path.join(images_out, out_rel), quality=90)
                    crop_records.append((out_rel, label))

    by_label: Dict[int, List[Tuple[str, int]]] = defaultdict(list)
    for path, label in crop_records:
        by_label[label].append((path, label))

    raw_a, raw_b = len(by_label[0]), len(by_label[1])
    if raw_a == 0 or raw_b == 0:
        raise RuntimeError(
            f"Not enough crops after size filter: car={raw_a}, van={raw_b}"
        )
    cap = min(CROP_MAX_PER_CLASS, raw_a, raw_b)

    selected: List[Tuple[str, int]] = []
    for label in (0, 1):
        pool = by_label[label][:]
        random.shuffle(pool)
        selected.extend(pool[:cap])

    selected_paths = {p for p, _ in selected}
    for root, _, files in os.walk(images_out):
        for fn in files:
            rel = os.path.relpath(os.path.join(root, fn), images_out).replace("\\", "/")
            if rel not in selected_paths:
                try:
                    os.remove(os.path.join(root, fn))
                except OSError:
                    pass

    rows: List[Tuple[str, int, str]] = []
    for label in (0, 1):
        pool = [(p, l) for p, l in selected if l == label]
        random.shuffle(pool)
        n = len(pool)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)
        train_part = pool[:n_train]
        val_part = pool[n_train : n_train + n_val]
        test_part = pool[n_train + n_val :]
        rows.extend((p, l, "train") for p, l in train_part)
        rows.extend((p, l, "validation") for p, l in val_part)
        rows.extend((p, l, "test") for p, l in test_part)

    counts = _write_csvs(out_dir, rows)
    names_title = [CLASS_A.capitalize(), CLASS_B.capitalize()]
    with open(os.path.join(out_dir, "class_names.txt"), "w", encoding="utf-8") as f:
        f.write(f"{names_title[0]}\n{names_title[1]}\n")

    n_a = sum(1 for _, l in selected if l == 0)
    n_b = sum(1 for _, l in selected if l == 1)
    print(f"  raw after filter: car={raw_a} van={raw_b}")
    print(f"  after balanced cap ({cap}/class): car={n_a} van={n_b} total={len(selected)}")
    for split in ("train", "validation", "test"):
        subset = [(p, l) for p, l, s in rows if s == split]
        print(f"  {split}: {len(subset)}  ({_class_hist(subset, names_title)})")
    print(f"  csv counts: {counts}")

    stats = {
        "raw_a": raw_a,
        "raw_b": raw_b,
        "n_a": n_a,
        "n_b": n_b,
        "total": len(selected),
        "cap": cap,
        "rows": rows,
        "counts": counts,
    }
    write_readme(out_dir, stats)
    print(f"  wrote {out_dir}")
    return stats


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate MAVREC car vs van crop subset from the val split."
    )
    p.add_argument(
        "--mavrec-parent",
        "--dataset-root",
        dest="mavrec_parent",
        default=None,
        help=(
            "Folder that contains MAVREC/ (input). "
            "Default: parent of scripts/."
        ),
    )
    p.add_argument(
        "--output-root",
        default=None,
        help=(
            "Where to write mavrec_car_van/. "
            "Default: <repo>/datasets (created if missing). "
            "Repo root is the parent of scripts/."
        ),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    repo_root = os.path.dirname(script_dir)

    mavrec_parent = os.path.abspath(args.mavrec_parent or repo_root)
    output_root = os.path.abspath(
        args.output_root or os.path.join(repo_root, "datasets")
    )
    os.makedirs(output_root, exist_ok=True)

    print(f"mavrec-parent: {mavrec_parent}")
    print(f"output-root:   {output_root}")

    build_car_van(mavrec_parent, output_root)

    print("\nDone.")


if __name__ == "__main__":
    main()
