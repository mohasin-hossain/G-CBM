"""Build single-node concept-bottleneck graphs for MLP/Linear-CBM ablations.

Each image is one node with feat z in R^K: max over patches of τ-masked NMF
scores. Same τ masking as gcbm_graph, but no CNN-weighted node features and
no inter-concept edges (DGL is used only for GraphDataLoader parity).
"""

from __future__ import annotations

import os
from typing import List, Optional

import dgl
import numpy as np
import torch
from dgl.data import DGLDataset

from .concepts import build_model_parts, load_craft_and_attach
from .utils import _safe_argmax

# Subdir under ``<output_root>/<dataset>/`` (parallel to ``graphs/<dataset>/``).
GRAPHS_SUBDIR = "graphs_concept_bottleneck_mlp_linear"


class ConceptBottleneckVectorDataset(DGLDataset):
    """One 1-node graph per image: feat = z ∈ R^K_+ (max-pooled masked U)."""

    def __init__(
        self,
        images,
        y,
        masks,
        patch_size,
        craft_xai,
        ignore_list,
        device,
        stride_r=0.5,
        coverage_threshold=0.5,
        seed=42,
        sim_threshold: float = 0.0,
    ):
        self.images = images
        self.y = y
        self.masks = masks
        self.patch_size = patch_size
        self.craft_xai = craft_xai
        self.ignore_list = ignore_list or []
        self.device = device
        self.stride_r = stride_r
        self.seed = seed
        self.coverage_threshold = coverage_threshold
        self.sim_threshold = float(sim_threshold)
        super().__init__(name="concept_bottleneck_mlp_linear_dataset")

    def _batch_inference(self, model, x, resize=None, device="cuda"):
        with torch.no_grad():
            x = x.clone().detach().to(device)
            if resize:
                x = torch.nn.functional.interpolate(
                    x, size=resize, mode="bicubic", align_corners=False
                )
            activation = model(x).cpu()
        return activation

    def process(self):
        self.graphs = []
        self.labels = []

        strides = int(self.patch_size * self.stride_r)
        if self.masks is None:
            self.masks = [None] * self.images.shape[0]

        for img, y, mask in zip(self.images, self.y, self.masks):
            img = img.unsqueeze(0)
            image_size = img.shape[2]

            if mask is None:
                patches = torch.nn.functional.unfold(
                    img, kernel_size=self.patch_size, stride=strides
                )
                patches = patches.transpose(1, 2).contiguous().view(
                    -1, img.shape[1], self.patch_size, self.patch_size
                )
            else:
                mask = mask.unsqueeze(0)
                img_patches = torch.nn.functional.unfold(
                    img, kernel_size=self.patch_size, stride=strides
                )
                img_patches = img_patches.transpose(1, 2).contiguous().view(
                    -1, 3, self.patch_size, self.patch_size
                )
                mask_patches = torch.nn.functional.unfold(
                    mask, kernel_size=self.patch_size, stride=strides
                )
                mask_patches = mask_patches.transpose(1, 2).contiguous().view(
                    -1, 1, self.patch_size, self.patch_size
                )
                coverage = mask_patches.float().mean(dim=(1, 2, 3))
                keep_indices = coverage >= self.coverage_threshold
                patches = img_patches[keep_indices]

            if patches.shape[0] == 0:
                continue

            self.craft_xai.device = self.device
            patch_activations = self._batch_inference(
                self.craft_xai.input_to_latent,
                patches,
                resize=image_size,
                device=self.device,
            )
            if len(patch_activations.shape) == 4:
                patch_activations = torch.mean(patch_activations, dim=(2, 3))

            W_dtype = self.craft_xai.reducer.components_.dtype
            patches_U = self.craft_xai.reducer.transform(
                np.array(patch_activations, dtype=W_dtype)
            )
            _ = _safe_argmax(patches_U, self.ignore_list)

            patches_U = torch.tensor(
                patches_U, dtype=torch.float32, device=self.device
            )
            if self.sim_threshold > 0.0:
                patches_U = patches_U * (
                    patches_U >= self.sim_threshold
                ).float()

            # z_c = max_j Û_{j,c}
            z = patches_U.max(dim=0).values  # [K]
            K = z.shape[0]
            for idx in self.ignore_list:
                if 0 <= idx < K:
                    z[idx] = 0.0

            g = dgl.graph(([0], [0]), num_nodes=1).to(self.device)
            g.ndata["feat"] = z.unsqueeze(0)
            self.graphs.append(g)
            self.labels.append(y)


def build_and_save_graphs_per_split(
    images: torch.Tensor,
    labels: torch.Tensor,
    device: str,
    backbone_name: str,
    craft_path: str,
    out_path: str,
    patch_size: int,
    stride_r: float,
    ignore_list: Optional[List[int]] = None,
    coverage_threshold: float = 0.0,
    sim_threshold: float = 0.0,
    backbone_weights: Optional[str] = None,
    **_unused_kwargs,
):
    from .concepts import resolve_backbone_weights
    ignore_list = ignore_list or []
    bw = resolve_backbone_weights(backbone_weights, craft_path)
    g, h = build_model_parts(
        backbone_name, device=device, pretrained=True, backbone_weights=bw)
    craft = load_craft_and_attach(craft_path, g, h)

    ds = ConceptBottleneckVectorDataset(
        images=images.to(device),
        y=labels.to(device),
        masks=None,
        patch_size=patch_size,
        craft_xai=craft,
        ignore_list=ignore_list,
        device=device,
        stride_r=stride_r,
        coverage_threshold=coverage_threshold,
        seed=42,
        sim_threshold=sim_threshold,
    )
    ds.process()

    graphs = ds.graphs
    labels_out = (
        torch.stack(ds.labels)
        if ds.labels and isinstance(ds.labels[0], torch.Tensor)
        else torch.tensor(ds.labels)
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    dgl.save_graphs(out_path, graphs, {"labels": labels_out})
    return out_path, len(graphs)


class LoadConceptGraphDataset(DGLDataset):
    def __init__(self, file_path=None, efeats=True, device="cuda"):
        self.file_path = file_path
        self.device = device
        self.efeats = efeats
        super().__init__(name="concept_bottleneck_loader")

    def load(self):
        self.graphs, metadata = dgl.load_graphs(self.file_path)
        self.labels = metadata["labels"]
        print(
            f"Loaded {len(self.graphs)} concept-bottleneck graphs from {self.file_path}"
        )

    def process(self):
        if self.file_path:
            self.load()
        else:
            self.graphs = []
            self.labels = []

    def __getitem__(self, idx):
        return self.graphs[idx], self.labels[idx]

    def __len__(self):
        return len(self.graphs)


def load_split(output_root: str, dataset: str, split: str, device: str = "cuda"):
    split_file = "validation" if split == "val" else split
    path = os.path.join(
        output_root,
        dataset,
        GRAPHS_SUBDIR,
        dataset,
        f"concept_graphs_{split_file}.dgl",
    )
    ds = LoadConceptGraphDataset(file_path=path, device=device)
    ds.load()
    return ds


def infer_dims(ds: LoadConceptGraphDataset):
    in_dim = ds.graphs[0].ndata["feat"].shape[1]
    num_classes = int(ds.labels.max().item()) + 1
    return in_dim, num_classes, None
