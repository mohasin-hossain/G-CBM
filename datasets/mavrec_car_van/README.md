# MAVREC crop subset (Car vs Van)

Binary classification subset for G-CBM-style pipelines, built by cropping COCO bounding boxes from the MAVREC val split.

## Attribution / licence

Adapted from **MAVREC** (CC BY 4.0). Credit the creators:

- **Title:** Multiview Aerial Visual RECognition (MAVREC): Can Multi-view Improve Aerial Visual Perception?
- **Authors:** Aritra Dutta, Srijan Das, Jacob Nielsen, Rajatsubhra Chakraborty, Mubarak Shah (CVPR 2024)
- **Licence:** [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)
- **Upstream:** [mavrec.github.io](https://mavrec.github.io/), [Hugging Face](https://huggingface.co/datasets/rjccv/MAVREC)

Full attribution text and BibTeX: see the repository [NOTICE](../../NOTICE).

```bibtex
@InProceedings{Dutta_2024_CVPR,
    author = {Dutta, Aritra and Das, Srijan and Nielsen, Jacob and Chakraborty, Rajatsubhra and Shah, Mubarak},
    title = {Multiview Aerial Visual RECognition (MAVREC): Can Multi-view Improve Aerial Visual Perception?},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month = {June},
    year = {2024},
    pages = {22678-22690}
}
```

## Source dataset

- **Hugging Face:** [https://huggingface.co/datasets/rjccv/MAVREC](https://huggingface.co/datasets/rjccv/MAVREC)
- **Google Drive (full data):** see `MAVREC/ACCESS_INSTRUCTIONS.md` after HF access is approved
- **Local source:**
  - Images: `MAVREC/data/labelled/val/{aerial,ground}/`
  - Annotations: `MAVREC/data/labelled/supervised_annotations.zip` (`aerial_valid.json`, `ground_val.json`)
- **Classes:** Car (COCO `car` → label **0**), Van (COCO `van` → label **1**)
- **Filter / processing:**
  - Pilot subset uses **official val only** (`train.zip` not required)
  - Full frames are multi-object; samples are **pre-cropped boxes** (not full images)
  - Skip boxes with width or height < **32** px; pad bbox by **5%**
  - Crops resized to max side **256**, saved as JPEG under `images/car/` and `images/van/`
  - Cap at **552** crops per class after shuffle (seed 42); raw before cap: car=9575, van=552
- **Images:** `images/car/` (552), `images/van/` (552); total **1104**



## CSVs (CBM-GAT format)


| File                    | Count | Per class           |
| ----------------------- | ----- | ------------------- |
| `train.csv` / `nmf.csv` | 772   | Car=386, Van=386    |
| `validation.csv`        | 166   | Car=83, Van=83      |
| `test.csv`              | 166   | Car=83, Van=83      |
| `all.csv`               | 1104  | with `split` column |


**Split rule:** per-class stratified random split (~70/15/15 train/val/test, seed 42) so each split stays class-balanced on this val-only pilot.  
`nmf.csv` is the same set as `train.csv`. All CSV paths resolve to real files under `images/`.

## Notes

- This is a **pilot** subset from val only. Rebuild from train+val when `train.zip` is available for a full experiment.
- Regenerator (from repo root): `python scripts/generate_mavrec_vehicle_crops.py --mavrec-parent /path/to/folder_with_MAVREC` — see [`scripts/README.md`](../../scripts/README.md).

