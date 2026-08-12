"""G-CBM concept explanations for one image.

Writes ``output_concept_explanation.png`` (localisation, importance bars,
exemplars) and ``output_active_patches.png`` under the repository root.
"""

from __future__ import annotations

import argparse
import os
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib as mpl
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from gcbm.concepts import (
    build_model_parts,
    load_craft_and_attach,
    resolve_backbone_weights,
)
from gcbm.config import (
    DATASETS,
    _repo_root,
    default_output_dir,
    get_class_label,
    get_dataset_params,
    resolve_under_repo,
)
from gcbm.gcbm_graph import ConceptGraphDataset, infer_dims, load_split
from gcbm.gcbm_model import EGATClassifier, GAT_LightningModule

mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
})


def _disp_label(dataset_key: str, idx: int) -> str:
    raw = get_class_label(dataset_key, idx)
    return raw.replace("Atypical/", "").replace("Common ", "")


def _label_color(label: str) -> str:
    l = label.lower()
    if any(w in l for w in ("nevus", "benign", "melanocytic", "normal")):
        return "#2ecc71"
    return "#e74c3c"


def _two_tone_caption(ax, y: float, prefix: str, label: str, label_color: str,
                      conf: Optional[float] = None, fontsize: int = 11):
    """Render ``prefix Label (xx.x%)`` with a coloured class name."""
    ax.text(0.5, y, f"{prefix} ",
            ha="right", va="top", fontsize=fontsize, fontweight="bold",
            color="black", transform=ax.transAxes)
    ax.text(0.5, y, label,
            ha="left", va="top", fontsize=fontsize, fontweight="bold",
            color=label_color, transform=ax.transAxes)
    if conf is not None:
        char_w = fontsize * 0.0026
        x_pct = 0.5 + (len(label) + 2.0) * char_w
        ax.text(x_pct, y, f"({conf * 100:.1f}%)",
                ha="left", va="top", fontsize=fontsize, fontweight="bold",
                color="black", transform=ax.transAxes)


def annotate_concept_patch_panel(
    image_pil: Image.Image,
    patches_U: np.ndarray,
    patch_importance_np: np.ndarray,
    top_concepts: List[int],
    colors: np.ndarray,
    stride: int,
    num_patches_w: int,
    patch_size: int,
    sorted_patch_idx: List[int],
    top_k: int,
) -> Image.Image:
    """One box per top concept: argmax of (concept activation × patch importance)."""
    img = image_pil.copy()
    draw = ImageDraw.Draw(img)
    num_concepts_total = patches_U.shape[1]

    selected_indices = []
    for concept_id in top_concepts:
        if concept_id >= num_concepts_total:
            continue
        scores = patches_U[:, concept_id] * patch_importance_np
        selected_indices.append((concept_id, int(np.argmax(scores))))
    if not selected_indices:
        selected_indices = [(top_concepts[0], idx) for idx in sorted_patch_idx[:top_k]]

    for concept_id, idx in selected_indices:
        row = idx // num_patches_w
        col = idx % num_patches_w
        x, y = col * stride, row * stride
        c_index = top_concepts.index(concept_id) if concept_id in top_concepts else 0
        outline_color = tuple((colors[c_index] * 255).astype(int))
        draw.rectangle(
            [x, y, x + patch_size, y + patch_size],
            outline=outline_color,
            width=3,
        )
    return img


def _greedy_distinct_patch_indices(
    scores_1d: np.ndarray,
    num_patches_h: int,
    num_patches_w: int,
    max_boxes: int,
    min_sep: int,
    floor_frac: float,
) -> List[int]:
    """Greedy NMS on the patch grid (Chebyshev distance)."""
    scores_1d = np.asarray(scores_1d, dtype=np.float64).reshape(-1)
    n = int(num_patches_h * num_patches_w)
    if scores_1d.size < n:
        pad = np.zeros(n, dtype=np.float64)
        pad[: scores_1d.size] = scores_1d
        scores_1d = pad
    elif scores_1d.size > n:
        scores_1d = scores_1d[:n]

    smax = float(scores_1d.max())
    if smax <= 1e-12:
        return [int(np.argmax(scores_1d))]

    chosen: List[int] = []
    for idx in np.argsort(-scores_1d):
        idx = int(idx)
        if scores_1d[idx] < floor_frac * smax:
            break
        r, c = idx // num_patches_w, idx % num_patches_w
        if all(
            max(abs(r - (j // num_patches_w)), abs(c - (j % num_patches_w))) >= min_sep
            for j in chosen
        ):
            chosen.append(idx)
            if len(chosen) >= max_boxes:
                break
    if not chosen:
        chosen = [int(np.argmax(scores_1d))]
    return chosen


def save_active_patches_row(
    top_concepts: List[int],
    colors: np.ndarray,
    patches_U: np.ndarray,
    patch_importance: torch.Tensor,
    num_patches_h: int,
    num_patches_w: int,
    image_pil: Image.Image,
    stride: int,
    patch_size: int,
    max_distinct_patches: int,
    min_patch_separation: int,
    floor_score_frac: float,
) -> str:
    """Most active patches per top concept (raw image + multi-boxes). PNG only."""
    patch_importance_np = patch_importance.detach().cpu().numpy()
    top_k = len(top_concepts)
    fig_w = max(12, 4 * top_k)
    fig, axes = plt.subplots(1, top_k, figsize=(fig_w, 4.5))
    if top_k == 1:
        axes = [axes]

    for i, concept_id in enumerate(top_concepts):
        scores = patches_U[:, concept_id] * patch_importance_np
        pil_img = image_pil.copy()
        draw = ImageDraw.Draw(pil_img)
        outline_color = tuple((colors[i] * 255).astype(int))
        for idx in _greedy_distinct_patch_indices(
            scores, num_patches_h, num_patches_w,
            max_distinct_patches, min_patch_separation, floor_score_frac,
        ):
            row, col = idx // num_patches_w, idx % num_patches_w
            x0, y0 = col * stride, row * stride
            draw.rectangle(
                [x0, y0, x0 + patch_size, y0 + patch_size],
                outline=outline_color,
                width=3,
            )
        ax = axes[i]
        ax.imshow(np.asarray(pil_img))
        ax.set_title(
            f"Concept {concept_id}",
            fontsize=14, fontweight="bold",
            color=tuple(colors[i][:3]),
        )
        ax.axis("off")

    fig.suptitle("Most active patches per concept",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    out_path = os.path.join(_repo_root, "output_active_patches.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print(f"[INFO] Active-patches figure saved to {out_path}")
    return out_path


def load_eval_transform(dataset_key: str):
    return DATASETS[dataset_key].build_transforms()["eval"]


def load_craft(dataset_key: str, device: str, output_root: str,
               backbone: str = "resnet50",
               backbone_weights: Optional[str] = None):
    """Rebuild g/h and attach to the saved light Craft.

    When ``backbone_weights`` is unset, falls back to
    ``craft/<ds>/backbone_weights.json`` written at CRAFT fit time.
    Otherwise uses ImageNet / torchvision defaults.
    """
    craft_dir = os.path.join(output_root, dataset_key, "craft", dataset_key)
    craft_path = os.path.join(craft_dir, f"craft_{dataset_key}.dill")
    if not os.path.isfile(craft_path):
        raise FileNotFoundError(f"Craft not found: {craft_path}")
    bw = resolve_backbone_weights(backbone_weights, craft_path)
    g, h = build_model_parts(
        backbone_name=backbone, device=device, pretrained=True,
        backbone_weights=bw,
    )
    return load_craft_and_attach(craft_path, g, h), craft_dir


def load_trained_gat(dataset_key: str, device: str, output_root: str,
                     in_dim: int, num_classes: int):
    ds_params = get_dataset_params(dataset_key) or {}
    model = EGATClassifier(
        in_feats=in_dim,
        out_feats=ds_params.get("hidden_dim", 128),
        num_heads=ds_params.get("num_heads", 4),
        out_dim=num_classes,
        feat_drop=0.0,
        node_drop=0.0,
    )
    ckpt = os.path.join(
        output_root, dataset_key, "models", dataset_key, f"{dataset_key}_best_model.ckpt")
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")
    return GAT_LightningModule.load_from_checkpoint(ckpt, model=model).model.eval().to(device)


@torch.no_grad()
def build_graph_from_single_image(dataset_key: str, image_path: str, device: str,
                                  craft, patch_size: int, stride_r: float):
    eval_tfm = load_eval_transform(dataset_key)
    image_pil = Image.open(image_path).convert("RGB").resize((224, 224), Image.BICUBIC)
    x = eval_tfm(image_pil).unsqueeze(0)
    ds = ConceptGraphDataset(
        images=x.to(device),
        y=torch.tensor([0], dtype=torch.long, device=device),
        masks=None,
        patch_size=patch_size,
        craft_xai=craft,
        ignore_list=[],
        device=device,
        stride_r=stride_r,
        coverage_threshold=0.0,
        seed=42,
        requires_grad=False,
    )
    ds.process()
    return ds[0][0].to(device), ds.patches_U, image_pil


def _compute_concepts(dataset_key: str, image_path: str, device: str,
                      backbone: str, output_root: str,
                      patch_size: int, stride_r: float,
                      top_k_max: int, min_concept_weight: float,
                      backbone_weights: Optional[str] = None):
    """Forward one image through G-CBM; return concept ranking + patch scores."""
    train_ds = load_split(output_root, dataset_key, "train", device=device)
    in_dim, num_classes, _ = infer_dims(train_ds)

    craft, craft_dir = load_craft(
        dataset_key, device, output_root, backbone=backbone,
        backbone_weights=backbone_weights)
    gat_model = load_trained_gat(dataset_key, device, output_root, in_dim, num_classes)
    graph, patches_U, image_pil = build_graph_from_single_image(
        dataset_key, image_path, device, craft, patch_size, stride_r)

    node_f = graph.ndata["feat"].float().to(device).requires_grad_(True)
    logits, _, _ = gat_model(graph, node_f)
    probs = F.softmax(logits[0], dim=0)
    pred_idx = int(torch.argmax(probs).item())
    pred_conf = float(probs[pred_idx].item())

    grads = torch.autograd.grad(probs[pred_idx], node_f, create_graph=False)[0]
    node_importance = grads.abs().sum(dim=1)
    node_importance = node_importance / (node_importance.sum() + 1e-8)

    sorted_values, sorted_indices = torch.sort(node_importance, descending=True)
    concept_importance_values = sorted_values.tolist()
    concept_ranking = sorted_indices.tolist()

    top_vals = [v for v in concept_importance_values if v > min_concept_weight]
    top_k = min(len(top_vals), top_k_max) if top_vals else min(top_k_max, len(concept_ranking))
    top_concepts = concept_ranking[:top_k]
    top_values = concept_importance_values[:top_k]

    U = torch.tensor(patches_U, device=node_importance.device, dtype=node_importance.dtype)
    patch_importance = torch.matmul(U, node_importance)
    patch_importance = patch_importance / (patch_importance.sum() + 1e-8)
    _, sorted_patch_idx = torch.sort(patch_importance, descending=True)

    colors = plt.cm.tab10(np.arange(len(top_concepts)) % 10)
    stride = int(patch_size * stride_r)
    num_patches_w = (image_pil.width - patch_size) // stride + 1
    num_patches_h = (image_pil.height - patch_size) // stride + 1

    return (
        image_pil, craft_dir, patch_importance, patches_U,
        top_concepts, top_values, sorted_patch_idx.tolist(),
        colors, stride, num_patches_w, num_patches_h,
        pred_idx, pred_conf,
    )


def explain_concepts(
    dataset_key: str,
    image_path: str,
    device: str,
    backbone: str,
    output_root: str,
    patch_size: int,
    stride_r: float,
    top_k_max: int,
    min_concept_weight: float,
    *,
    true_class: int = -1,
    max_distinct_patches_per_concept: int = 4,
    min_patch_separation: int = 3,
    floor_score_frac: float = 0.15,
    backbone_weights: Optional[str] = None,
):
    """Write explanation and active-patch PNGs for one image."""
    (image_pil, craft_dir, patch_importance, patches_U,
     top_concepts, top_values, sorted_patch_idx,
     colors, stride, num_patches_w, num_patches_h,
     pred_idx, pred_conf) = _compute_concepts(
        dataset_key, image_path, device, backbone, output_root,
        patch_size, stride_r, top_k_max, min_concept_weight,
        backbone_weights=backbone_weights,
    )
    top_k = len(top_concepts)
    patch_importance_np = patch_importance.detach().cpu().numpy()

    image_annot = annotate_concept_patch_panel(
        image_pil, patches_U, patch_importance_np, top_concepts, colors,
        stride, num_patches_w, patch_size, sorted_patch_idx, top_k,
    )

    pred_short = _disp_label(dataset_key, pred_idx)
    pred_color = _label_color(pred_short)
    show_gt = true_class >= 0
    if show_gt:
        gt_short = _disp_label(dataset_key, true_class)
        gt_color = _label_color(gt_short)

    def _draw_caption_gt(ax):
        ax.axis("off")
        if show_gt:
            _two_tone_caption(ax, 0.80, "Ground truth:", gt_short, gt_color)

    def _draw_caption_pred(ax):
        ax.axis("off")
        _two_tone_caption(ax, 0.80, "Predicted:", pred_short, pred_color, conf=pred_conf)

    # 3-panel layout: localisation | spacer | bars | exemplars
    cap_ratio = 3
    fig = plt.figure(figsize=(14, 6))
    outer_gs = gridspec.GridSpec(
        1, 4, width_ratios=[1, 0.10, 1, 1],
        wspace=0.12, left=0.02, right=0.98, top=0.94, bottom=0.04,
    )

    def _col_subgs(col_idx):
        return gridspec.GridSpecFromSubplotSpec(
            3, 1, subplot_spec=outer_gs[0, col_idx],
            height_ratios=[2, 28, cap_ratio], hspace=0.04)

    # (a) Concept localisation
    gs_a = _col_subgs(0)
    ax_a_title = fig.add_subplot(gs_a[0, 0]); ax_a_title.axis("off")
    ax_a_title.text(0.5, 0.5, "Concept Localisation",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_a_title.transAxes)
    ax_cbm = fig.add_subplot(gs_a[1, 0])
    ax_cbm.imshow(np.array(image_annot)); ax_cbm.axis("off")
    _draw_caption_gt(fig.add_subplot(gs_a[2, 0]))

    # (b) Importance bars
    gs_b = _col_subgs(2)
    ax_b_title = fig.add_subplot(gs_b[0, 0]); ax_b_title.axis("off")
    ax_b_title.text(0.5, 0.5, "Top 3 Concept Activations",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_b_title.transAxes)
    ax_bar = fig.add_subplot(gs_b[1, 0])
    x_pos = np.arange(top_k)
    bars = ax_bar.bar(x_pos, top_values, color=colors[:top_k], align="center", width=0.55)
    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels([f"Concept {c}" for c in top_concepts], fontsize=11)
    ax_bar.set_ylabel("Gradient importance weight", fontsize=12)
    ax_bar.set_box_aspect(1)
    if top_k:
        ax_bar.set_ylim(0, max(top_values) * 1.25)
        for i, b in enumerate(bars):
            ax_bar.text(
                b.get_x() + b.get_width() / 2,
                b.get_height() + max(top_values) * 0.015,
                f"{top_values[i]:.2f}",
                ha="center", va="bottom", fontsize=12, fontweight="bold",
            )
    _draw_caption_pred(fig.add_subplot(gs_b[2, 0]))

    # (c) Exemplar thumbnails
    gs_c = _col_subgs(3)
    ax_c_title = fig.add_subplot(gs_c[0, 0]); ax_c_title.axis("off")
    ax_c_title.text(0.5, 0.5, "Top 3 Concept Exemplars",
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    transform=ax_c_title.transAxes)
    right_gs = gridspec.GridSpecFromSubplotSpec(
        top_k, 2, subplot_spec=gs_c[1, 0],
        width_ratios=[0.1, 1.0], wspace=0.0, hspace=0.12)
    for i in range(top_k):
        concept_id = top_concepts[i]
        c_color = colors[i]
        ax_c = fig.add_subplot(right_gs[i, 1])
        ax_c.axis("off")
        thumb = os.path.join(craft_dir, "concept_examples", f"concept_{concept_id}.png")
        if os.path.isfile(thumb):
            ax_c.imshow(Image.open(thumb).convert("RGB"))
            ax_c.add_patch(mpatches.Rectangle(
                (0, 0), 1, 1, transform=ax_c.transAxes,
                linewidth=8, edgecolor=c_color, facecolor="none"))
        else:
            ax_c.text(0.5, 0.5, f"(no example for {concept_id})",
                      ha="center", va="center", fontsize=11)
        ax_c.text(0.5, 1.07, f"Concept {concept_id}",
                  ha="center", va="bottom", fontsize=12, fontweight="bold",
                  color=c_color, transform=ax_c.transAxes)
        ax_c.text(0.5, -0.05, f"Importance: {top_values[i] * 100:.1f}%",
                  ha="center", va="top", fontsize=11, transform=ax_c.transAxes)
    fig.add_subplot(gs_c[2, 0]).axis("off")

    out_main = os.path.join(_repo_root, "output_concept_explanation.png")
    plt.savefig(out_main, dpi=300, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    print(f"[INFO] Concept-explanation figure saved to {out_main}")

    save_active_patches_row(
        top_concepts, colors, patches_U, patch_importance,
        num_patches_h, num_patches_w, image_pil, stride, patch_size,
        max_distinct_patches_per_concept, min_patch_separation, floor_score_frac,
    )

    print(f"G-CBM -> {get_class_label(dataset_key, pred_idx)} ({pred_conf * 100:.2f}%)")
    print(f"Top concepts: {', '.join(str(c) for c in top_concepts)}")


def main():
    ap = argparse.ArgumentParser(
        description="G-CBM concept explanation figures")
    ap.add_argument("--dataset", choices=list(DATASETS.keys()), required=True)
    ap.add_argument("--image_path", required=True, type=str)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--backbone", default="resnet50")
    ap.add_argument(
        "--backbone-weights", default=None,
        help="Optional fine-tuned CNN .pt used when CRAFT was fit. "
             "Default: read craft/*/backbone_weights.json if present, else "
             "ImageNet / torchvision weights.",
    )
    ap.add_argument("--output-root", default=default_output_dir,
                    help="Root containing craft / graphs / models")
    ap.add_argument("--patch-size", type=int, default=70)
    ap.add_argument("--stride-r", type=float, default=0.5)
    ap.add_argument("--top-k-max", type=int, default=3)
    ap.add_argument("--min-concept-weight", type=float, default=0.01)
    ap.add_argument(
        "--true-class", type=int, default=-1,
        help="Ground-truth class index (0-based); -1 omits the GT caption.",
    )
    ap.add_argument("--max-distinct-patches-per-concept", type=int, default=4)
    ap.add_argument("--min-patch-separation", type=int, default=3)
    ap.add_argument("--floor-score-frac", type=float, default=0.15)
    args = ap.parse_args()

    args.output_root = resolve_under_repo(args.output_root)

    explain_concepts(
        dataset_key=args.dataset,
        image_path=args.image_path,
        device=args.device,
        backbone=args.backbone,
        output_root=args.output_root,
        patch_size=args.patch_size,
        stride_r=args.stride_r,
        top_k_max=args.top_k_max,
        min_concept_weight=args.min_concept_weight,
        true_class=args.true_class,
        max_distinct_patches_per_concept=args.max_distinct_patches_per_concept,
        min_patch_separation=args.min_patch_separation,
        floor_score_frac=args.floor_score_frac,
        backbone_weights=args.backbone_weights,
    )


if __name__ == "__main__":
    main()
