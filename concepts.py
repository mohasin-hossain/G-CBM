"""Backbone feature extractors, CRAFT/NMF fitting, and auto-K selection."""

import os
import dill
import json
from typing import List, Tuple, Dict, Optional
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import Counter
from craft.craft_torch import Craft, torch_to_numpy


def build_model_parts(backbone_name: str = "resnet50",
                      device: str = "cuda",
                      pretrained: bool = True) -> Tuple[nn.Module, nn.Module]:
    """Return (g, h): g maps images to a spatial feature map; h maps that map to logits.

    Supported: resnet18, resnet50, densenet201, mobilenet_v2.
    """
    backbone_name = backbone_name.lower()
    if backbone_name == "resnet18":
        from pytorchcv.model_provider import get_model as ptcv_get_model

        ptcv_root = os.path.join(
            os.environ.get("TORCH_HOME", os.path.expanduser("~/.torch")),
            "pytorchcv",
        )
        net = ptcv_get_model("resnet18_cub", pretrained=pretrained, root=ptcv_root)
        g = nn.Sequential(*list(net.features.children())[:-1]).to(device).eval()
        fc = net.output
        h = lambda x, _fc=fc: _fc(torch.mean(x, (2, 3)))
        return g, h

    elif backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        model = models.resnet50(weights=weights)
        g = nn.Sequential(*list(model.children())[:-2]).to(device).eval()
        fc = model.fc
        h = lambda x, _fc=fc: _fc(torch.mean(x, (2, 3)))
        return g, h

    elif backbone_name == "densenet201":
        import torch.nn.functional as _F
        weights = models.DenseNet201_Weights.DEFAULT if pretrained else None
        model = models.densenet201(weights=weights)
        # DenseNet features end in BatchNorm (can be negative); CRAFT NMF needs
        # non-negative maps, so append ReLU on g and use the same for h's input.
        g = nn.Sequential(model.features, nn.ReLU(inplace=False)).to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    elif backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        g = model.features.to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    else:
        raise ValueError(
            f"Unsupported backbone for Craft: {backbone_name!r}. "
            "Choose from: resnet18, resnet50, densenet201, mobilenet_v2"
        )


def fit_craft_for_k(images: torch.Tensor,
                    k: int,
                    patch_size: int,
                    batch_size: int,
                    device: str,
                    g: nn.Module,
                    h: nn.Module) -> Tuple[Craft, torch.Tensor]:
    """Fit CRAFT with k concepts; return (craft, crops, crops_u)."""
    craft = Craft(
        input_to_latent=g.to(device),
        latent_to_logit=h,
        number_of_concepts=k,
        patch_size=patch_size,
        batch_size=batch_size,
        device=device,
    )
    crops, crops_u, w = craft.fit(images.to(device))
    crops = np.moveaxis(torch_to_numpy(crops), 1, -1)
    return craft, crops, crops_u


def score_concepts_from_u(crops_u: torch.Tensor,
                          labels: torch.Tensor,
                          q: float = 0.1,
                          theta: float = 0.6,
                          lambda_weight: float = 1.0) -> Tuple[float, Dict]:
    """Discriminativeness score over concepts from patch activations crops_u.

    patches_per_image is inferred as crops_u.shape[0] // N_images.
    """
    if crops_u.size == 0:
        return -1.0, {"Avg D_i": 0.0, "Penalty": 1.0, "Num Discriminative Concepts": 0, "Class Split": "None"}

    U = crops_u
    num_patches, num_concepts = U.shape
    labels_np = labels.detach().cpu().numpy()
    N_images = len(labels_np)
    patches_per_image = max(1, num_patches // max(1, N_images))
    num_classes = len(set(labels_np)) if len(labels_np) > 0 else 0

    results = []
    for i in range(num_concepts):
        u_i = U[:, i]
        tau_i = np.quantile(u_i, 1 - q)
        top_patch_indices = np.where(u_i >= tau_i)[0]
        top_image_indices = top_patch_indices // patches_per_image
        top_class_labels = labels_np[top_image_indices]

        class_counts = Counter(top_class_labels)
        total = len(top_class_labels) if len(top_class_labels) > 0 else 1
        R_ic = [class_counts.get(c, 0) / total for c in range(num_classes)]
        D_i = max(R_ic) if R_ic else 0.0
        dominant_class = int(np.argmax(R_ic)) if R_ic else 0

        results.append({
            "concept": i,
            "D_i": D_i,
            "dominant_class": dominant_class if D_i >= theta else None
        })

    discriminative = [r for r in results if r["D_i"] >= theta]
    d = len(discriminative)

    if d > 0 and num_classes > 0:
        avg_Di = float(np.mean([r["D_i"] for r in discriminative]))
        class_assignments = [r["dominant_class"] for r in discriminative]
        class_counts = Counter(class_assignments)
        penalty = sum([abs(class_counts.get(c, 0) / d - 1 / num_classes) for c in range(num_classes)]) / num_classes
        score = avg_Di - lambda_weight * penalty
    else:
        avg_Di = 0.0
        penalty = 1.0
        score = -lambda_weight * penalty

    summary = {
        "Avg D_i": round(avg_Di, 4),
        "Penalty": round(penalty, 4),
        'Class Split': [class_counts.get(c, 0) for c in range(num_classes)],
        "Num Discriminative Concepts": d
    }
    return float(score), summary


def auto_select_k(images: torch.Tensor,
                  labels: torch.Tensor,
                  candidates: List[int],
                  patch_size: int,
                  batch_size: int,
                  device: str,
                  g: nn.Module,
                  h: nn.Module) -> Tuple[int, List[Dict], Craft]:
    """Pick k in candidates that maximises score_concepts_from_u."""
    best_k, best_score = None, -1e9
    table = []
    for k in candidates:
        _, _, crops_u = fit_craft_for_k(images, k, patch_size, batch_size, device, g, h)
        score, summary = score_concepts_from_u(crops_u, labels)
        rec = {"Num Concepts": k, **summary, "Score": round(score, 4)}
        table.append(rec)
        if score > best_score:
            best_score = score
            best_k = k
    return best_k, table


def save_craft_light(craft: Craft, out_path: str):
    """Serialize Craft without the backbone modules (reattach on load)."""
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    g = getattr(craft, "input_to_latent", None)
    h = getattr(craft, "latent_to_logit", None)
    try:
        craft.input_to_latent = None
        craft.latent_to_logit = None
        with open(out_path, "wb") as f:
            dill.dump(craft, f)
    finally:
        craft.input_to_latent = g
        craft.latent_to_logit = h


def load_craft_and_attach(path: str, g: nn.Module, h: nn.Module) -> Craft:
    """Load a light Craft dump and attach backbone modules g, h."""
    with open(path, "rb") as f:
        craft = dill.load(f)
    craft.input_to_latent = g
    craft.latent_to_logit = h
    return craft


def write_best_k(craft_dir: str, best_k: int, patch_size: int, stride_r: float):
    """Write U_meta/nmf_best_k.json with best_k, patch_size, stride_r."""
    u_meta = os.path.join(craft_dir, "U_meta")
    os.makedirs(u_meta, exist_ok=True)
    payload = {
        "best_k": int(best_k),
        "patch_size": int(patch_size),
        "stride_r": float(stride_r),
    }
    out_path = os.path.join(u_meta, "nmf_best_k.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)