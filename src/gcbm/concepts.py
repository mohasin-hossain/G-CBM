"""Backbone feature extractors, CRAFT/NMF fitting, and auto-K selection."""

import os
import dill
import json
from typing import List, Tuple, Dict, Optional, Any
import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np
from collections import Counter
from craft.craft_torch import Craft, torch_to_numpy


BACKBONE_WEIGHTS_META = "backbone_weights.json"


def _strip_module_prefix(state: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    if not state:
        return state
    if all(k.startswith(prefix) for k in state):
        n = len(prefix)
        return {k[n:]: v for k, v in state.items()}
    return state


def _extract_state_dict(ckpt: Any) -> Tuple[Dict[str, Any], Optional[int]]:
    """Parse train_cnn .pt / Lightning ckpt into (state_dict, num_classes?)."""
    num_classes = None
    if not isinstance(ckpt, dict):
        raise ValueError("Checkpoint must be a dict (state_dict or wrapped .pt).")

    if "num_classes" in ckpt and isinstance(ckpt["num_classes"], int):
        num_classes = int(ckpt["num_classes"])

    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
        state = ckpt["state_dict"]
    elif "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
    else:
        state = {k: v for k, v in ckpt.items() if torch.is_tensor(v)}
        if not state:
            raise ValueError("Could not find a tensor state_dict in checkpoint.")

    state = _strip_module_prefix(state, "model.")
    state = _strip_module_prefix(state, "module.")
    return state, num_classes


def _resize_classifier_for_state(model: nn.Module, backbone_name: str,
                                 state: Dict[str, Any],
                                 num_classes: Optional[int]) -> None:
    """Match classifier head size to checkpoint before load_state_dict."""
    if backbone_name == "resnet50":
        key = "fc.weight"
        if key in state:
            out_features = int(state[key].shape[0])
        elif num_classes is not None:
            out_features = num_classes
        else:
            return
        if model.fc.out_features != out_features:
            model.fc = nn.Linear(model.fc.in_features, out_features)
    elif backbone_name == "densenet201":
        key = "classifier.weight"
        if key in state:
            out_features = int(state[key].shape[0])
        elif num_classes is not None:
            out_features = num_classes
        else:
            return
        if model.classifier.out_features != out_features:
            model.classifier = nn.Linear(model.classifier.in_features, out_features)
    elif backbone_name == "mobilenet_v2":
        key = "classifier.1.weight"
        if key in state:
            out_features = int(state[key].shape[0])
        elif num_classes is not None:
            out_features = num_classes
        else:
            return
        if model.classifier[1].out_features != out_features:
            model.classifier[1] = nn.Linear(
                model.classifier[1].in_features, out_features)


def load_weights_into_backbone(model: nn.Module, backbone_name: str,
                               weights_path: str) -> None:
    """Load a fine-tuned CNN checkpoint (from train_cnn) into ``model``."""
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"backbone weights not found: {weights_path}")
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    state, num_classes = _extract_state_dict(ckpt)
    _resize_classifier_for_state(model, backbone_name, state, num_classes)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad_missing = [k for k in missing if not k.startswith(("fc.", "classifier."))]
    if bad_missing:
        raise RuntimeError(
            f"Failed loading backbone weights from {weights_path}: "
            f"missing keys {bad_missing[:8]}{'...' if len(bad_missing) > 8 else ''}"
        )
    if unexpected:
        print(f"[warn] unexpected keys when loading {weights_path}: "
              f"{unexpected[:8]}{'...' if len(unexpected) > 8 else ''}")
    print(f"[INFO] Loaded backbone weights from {weights_path}")


def write_backbone_weights_meta(craft_dir: str, weights_path: str,
                                backbone_name: str) -> str:
    """Persist which CNN checkpoint was used to fit CRAFT (for later rebuilds)."""
    os.makedirs(craft_dir, exist_ok=True)
    out = os.path.join(craft_dir, BACKBONE_WEIGHTS_META)
    payload = {
        "backbone_weights": os.path.abspath(weights_path),
        "backbone": backbone_name,
    }
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    return out


def read_backbone_weights_meta(craft_path_or_dir: str) -> Optional[str]:
    """Return absolute backbone_weights path from CRAFT meta, or None."""
    if not craft_path_or_dir:
        return None
    if os.path.isdir(craft_path_or_dir):
        meta = os.path.join(craft_path_or_dir, BACKBONE_WEIGHTS_META)
    else:
        meta = os.path.join(os.path.dirname(craft_path_or_dir), BACKBONE_WEIGHTS_META)
    if not os.path.isfile(meta):
        return None
    with open(meta, "r") as f:
        payload = json.load(f)
    path = payload.get("backbone_weights")
    if not path:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{meta} points to missing backbone weights: {path}"
        )
    return path


def resolve_backbone_weights(explicit: Optional[str],
                             craft_path_or_dir: Optional[str] = None
                             ) -> Optional[str]:
    """Prefer explicit CLI path; else CRAFT-side meta written at fit time."""
    if explicit:
        return explicit
    return read_backbone_weights_meta(craft_path_or_dir) if craft_path_or_dir else None


def build_model_parts(backbone_name: str = "resnet50",
                      device: str = "cuda",
                      pretrained: bool = True,
                      backbone_weights: Optional[str] = None
                      ) -> Tuple[nn.Module, nn.Module]:
    """Return (g, h): g maps images to a spatial feature map; h maps that map to logits.

    Supported: resnet50, densenet201, mobilenet_v2.

    backbone_weights:
      Optional path to a fine-tuned CNN checkpoint from ``train_cnn``
      (``{dataset}_{backbone}_cnn.pt``). When set, those weights replace the
      default ImageNet / torchvision init. When None (default), behaviour is
      unchanged.
    """
    backbone_name = backbone_name.lower()
    use_pretrained = bool(pretrained) and not backbone_weights

    if backbone_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT if use_pretrained else None
        model = models.resnet50(weights=weights)
        if backbone_weights:
            load_weights_into_backbone(model, backbone_name, backbone_weights)
        g = nn.Sequential(*list(model.children())[:-2]).to(device).eval()
        fc = model.fc
        h = lambda x, _fc=fc: _fc(torch.mean(x, (2, 3)))
        return g, h

    elif backbone_name == "densenet201":
        weights = models.DenseNet201_Weights.DEFAULT if use_pretrained else None
        model = models.densenet201(weights=weights)
        if backbone_weights:
            load_weights_into_backbone(model, backbone_name, backbone_weights)
        # DenseNet features end in BatchNorm (can be negative); CRAFT NMF needs
        # non-negative maps, so append ReLU on g and use the same for h's input.
        g = nn.Sequential(model.features, nn.ReLU(inplace=False)).to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    elif backbone_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT if use_pretrained else None
        model = models.mobilenet_v2(weights=weights)
        if backbone_weights:
            load_weights_into_backbone(model, backbone_name, backbone_weights)
        g = model.features.to(device).eval()
        classifier = model.classifier
        h = lambda x, _clf=classifier: _clf(x.mean([2, 3]))
        return g, h

    else:
        raise ValueError(
            f"Unsupported backbone for Craft: {backbone_name!r}. "
            "Choose from: resnet50, densenet201, mobilenet_v2"
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