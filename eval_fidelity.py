"""Evaluate G-CBM faithfulness via concept-node deletion and insertion.

Loads a trained checkpoint and its test graphs, ranks nodes by gradient
magnitude or randomly, and writes per-image CSVs plus aggregated JSON
curves (mean ± std over the test set).
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from config import DATASETS, get_dataset_params
from utils import _set_seed
from gcbm_model import EGATClassifier, GAT_LightningModule


def compute_alpha_nodegrad(device, gat_model, graph):
    """Per-node α = L1 of grad of predicted-class prob w.r.t. node features."""
    node_f_base = graph.ndata['feat'].float().to(device)
    node_f = node_f_base.clone().detach().requires_grad_(True)
    preds, _, _ = gat_model(graph, node_f)
    y_hat = preds[0].argmax().item()
    target_prob = F.softmax(preds[0], dim=0)[y_hat]
    grads = torch.autograd.grad(target_prob, node_f, create_graph=False)[0]
    alpha = grads.abs().sum(dim=1).detach()
    order_vals, order_idx = torch.sort(alpha, descending=True)
    return alpha, order_idx, node_f_base


@torch.no_grad()
def _forward_with_masked_nodef(gat_model, graph, node_f_base, keep_idx):
    """Zero all nodes except keep_idx, then forward."""
    nf = torch.zeros_like(node_f_base)
    if keep_idx.numel() > 0:
        nf[keep_idx] = node_f_base[keep_idx]
    logits, _, _ = gat_model(graph, nf)
    return logits.squeeze(0)  # (C,)


def _load_split(run_root: str, dataset: str, split: str, device: str):
    from gcbm_graph import load_split
    return load_split(run_root, dataset, split, device)


def _order_topk_grad(device, gat_model, graph):
    """Rank nodes by gradient magnitude w.r.t. the predicted class."""
    _, order_idx, node_f_base = compute_alpha_nodegrad(device, gat_model, graph)
    return order_idx, node_f_base


def _order_random(device, gat_model, graph, rng: np.random.Generator):
    """Uniform random node order (random ablation baseline)."""
    node_f_base = graph.ndata['feat'].float().to(device)
    r = node_f_base.size(0)
    perm = rng.permutation(r)
    order_idx = torch.as_tensor(perm, device=device, dtype=torch.long)
    return order_idx, node_f_base


def _fidelity_curve(device, gat_model, graph, order_idx, node_f_base,
                    frk_step: int):
    """Return (fracs, deletion_probs, insertion_probs, auc_del, auc_ins,
    p_full, p_empty) for a single graph at the given ranking."""
    r = node_f_base.size(0)

    logits_full, _, _ = gat_model(graph, node_f_base)
    y_hat = logits_full[0].argmax().item()
    p_full = F.softmax(logits_full[0], dim=0)[y_hat].item()
    logits_empty = _forward_with_masked_nodef(
        gat_model, graph, node_f_base,
        keep_idx=torch.tensor([], device=device, dtype=torch.long))
    p_empty = F.softmax(logits_empty, dim=0)[y_hat].item()

    fracs = np.linspace(0.0, 1.0, frk_step + 1)
    deletion_probs, insertion_probs = [], []

    for i in range(frk_step + 1):
        k = int(round(i * r / frk_step))

        keep_del = (order_idx[k:] if k < r
                    else torch.tensor([], device=device, dtype=torch.long))
        log_del = _forward_with_masked_nodef(
            gat_model, graph, node_f_base, keep_idx=keep_del)
        deletion_probs.append(F.softmax(log_del, dim=0)[y_hat].item())

        keep_ins = (order_idx[:k] if k > 0
                    else torch.tensor([], device=device, dtype=torch.long))
        log_ins = _forward_with_masked_nodef(
            gat_model, graph, node_f_base, keep_idx=keep_ins)
        insertion_probs.append(F.softmax(log_ins, dim=0)[y_hat].item())

    deletion_probs = np.asarray(deletion_probs)
    insertion_probs = np.asarray(insertion_probs)
    auc_del = float(np.trapz(deletion_probs, fracs))
    auc_ins = float(np.trapz(insertion_probs, fracs))

    return (fracs, deletion_probs, insertion_probs,
            auc_del, auc_ins, p_full, p_empty)


def _evaluate_one(device, gat_model, dataset, strategy: str,
                  frk_step: int, seed: int):
    rng = np.random.default_rng(seed)
    grid = np.linspace(0.0, 1.0, frk_step + 1)
    curves_del: List[np.ndarray] = []
    curves_ins: List[np.ndarray] = []
    per_image = []

    for index, graph_data in enumerate(dataset):
        graph = graph_data[0].to(device)
        try:
            if strategy == "topk_grad":
                order_idx, node_f_base = _order_topk_grad(
                    device, gat_model, graph)
            elif strategy == "random":
                order_idx, node_f_base = _order_random(
                    device, gat_model, graph, rng)
            else:
                raise ValueError(f"Unknown strategy: {strategy}")

            (fr, delp, insp, aucd, auci,
             p_full, p_empty) = _fidelity_curve(
                device, gat_model, graph, order_idx, node_f_base, frk_step)

            curves_del.append(np.asarray(delp, dtype=np.float64))
            curves_ins.append(np.asarray(insp, dtype=np.float64))
            per_image.append({
                "image_index": index,
                "AUC_del": aucd,
                "AUC_ins": auci,
                "p_full": p_full,
                "p_empty": p_empty,
            })
        except Exception as e:  # pragma: no cover — defensive
            print(f"[skip] {index}: {e}")

    n_ok = len(curves_del)
    if n_ok == 0:
        return {
            "n_images": 0,
            "fracs": grid.tolist(),
            "mean_deletion_probs": [],
            "mean_insertion_probs": [],
            "std_deletion_probs": [],
            "std_insertion_probs": [],
            "mean_AUC_del": float("nan"),
            "mean_AUC_ins": float("nan"),
            "std_AUC_del": float("nan"),
            "std_AUC_ins": float("nan"),
        }, per_image

    stack_del = np.stack(curves_del, axis=0)
    stack_ins = np.stack(curves_ins, axis=0)
    mean_del = stack_del.mean(axis=0).tolist()
    mean_ins = stack_ins.mean(axis=0).tolist()
    std_del = stack_del.std(axis=0, ddof=0).tolist()
    std_ins = stack_ins.std(axis=0, ddof=0).tolist()

    aucs_del = np.array([row["AUC_del"] for row in per_image])
    aucs_ins = np.array([row["AUC_ins"] for row in per_image])

    return {
        "n_images": n_ok,
        "fracs": grid.tolist(),
        "mean_deletion_probs": mean_del,
        "mean_insertion_probs": mean_ins,
        "std_deletion_probs": std_del,
        "std_insertion_probs": std_ins,
        "mean_AUC_del": float(aucs_del.mean()),
        "mean_AUC_ins": float(aucs_ins.mean()),
        "std_AUC_del": float(aucs_del.std()),
        "std_AUC_ins": float(aucs_ins.std()),
    }, per_image


def main():
    ap = argparse.ArgumentParser(
        "G-CBM concept-node deletion/insertion faithfulness")
    ap.add_argument("--run-root", required=True,
                    help="Path to the trained run concept_graph_data root")
    ap.add_argument("--datasets", nargs="+",
                    default=["ham10000", "ph2", "derm7pt", "imagenet"],
                    choices=list(DATASETS.keys()))
    ap.add_argument("--strategy", nargs="+",
                    default=["topk_grad", "random"],
                    choices=["topk_grad", "random"],
                    help="One or more ranking strategies. Each writes "
                         "its own JSON + CSV.")
    ap.add_argument("--frk-step", type=int, default=10,
                    help="Number of fraction steps; total grid points = "
                         "frk_step + 1. Default 10 → 11 points.")
    ap.add_argument("--out-dir", default=None,
                    help="Output directory. Defaults to "
                         "<run_root>/fidelity/gcbm/")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--num-heads", type=int, default=None)
    args = ap.parse_args()

    device = (args.device
              if (args.device.startswith("cuda") and torch.cuda.is_available())
              else "cpu")
    _set_seed(args.seed)

    out_dir = (args.out_dir or
               os.path.join(args.run_root, "fidelity", "gcbm"))
    os.makedirs(out_dir, exist_ok=True)

    for ds in args.datasets:
        print(f"\n==================== DATASET: {ds} (gcbm) ====================")

        ds_params = get_dataset_params(ds) or {}
        num_heads = (args.num_heads
                     if args.num_heads is not None
                     else ds_params.get("num_heads"))
        hidden_dim = (args.hidden_dim
                      if args.hidden_dim is not None
                      else ds_params.get("hidden_dim", args.hidden_dim))

        try:
            test_ds = _load_split(args.run_root, ds, "test", device)
        except FileNotFoundError as e:
            print(f"[skip dataset {ds}] graphs not found: {e}")
            continue

        in_dim = test_ds.graphs[0].ndata["feat"].shape[1]
        num_classes = int(test_ds.labels.max().item()) + 1

        model = EGATClassifier(
            in_feats=in_dim,
            out_feats=hidden_dim,
            num_heads=num_heads,
            out_dim=num_classes,
            feat_drop=0.0,
            node_drop=0.0,
        )

        ckpt = os.path.join(
            args.run_root, ds, "models", ds, f"{ds}_best_model.ckpt")
        if not os.path.isfile(ckpt):
            print(f"[skip dataset {ds}] checkpoint not found: {ckpt}")
            continue

        gat = GAT_LightningModule.load_from_checkpoint(ckpt, model=model)
        gat = gat.model.eval().to(device)

        for strategy in args.strategy:
            print(f"  strategy: {strategy}")
            agg, per_image = _evaluate_one(
                device, gat, test_ds, strategy, args.frk_step, args.seed)

            agg_payload = {
                "dataset": ds,
                "model": "gcbm",
                "strategy": strategy,
                "frk_step": args.frk_step,
                "seed": args.seed,
                "run_root": args.run_root,
                **agg,
            }

            json_path = os.path.join(out_dir, f"{ds}_{strategy}.json")
            csv_path = os.path.join(out_dir, f"{ds}_{strategy}.csv")
            with open(json_path, "w") as f:
                json.dump(agg_payload, f, indent=2)

            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=["image_index", "AUC_del", "AUC_ins",
                                "p_full", "p_empty"])
                writer.writeheader()
                for row in per_image:
                    writer.writerow(row)

            print(f"    n_images           : {agg['n_images']}")
            print(f"    mean AUC deletion  : {agg['mean_AUC_del']:.4f} "
                  f"± {agg['std_AUC_del']:.4f}")
            print(f"    mean AUC insertion : {agg['mean_AUC_ins']:.4f} "
                  f"± {agg['std_AUC_ins']:.4f}")
            print(f"    -> {json_path}")
            print(f"    -> {csv_path}")


if __name__ == "__main__":
    main()
