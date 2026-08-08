"""Shared dataset helpers, seeding, and concept exemplar utilities."""

import os
import random
from math import ceil

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class ImageDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data.iloc[idx, 0])
        label = self.data.iloc[idx, 1]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _tensors_from_loader(dl: DataLoader):
    xs, ys = [], []
    for x, y in dl:
        xs.append(x)
        ys.append(y)
    if xs:
        return torch.cat(xs, dim=0), torch.cat(ys, dim=0)
    return torch.empty(0, 3, 224, 224), torch.empty(0, dtype=torch.long)


model_preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _safe_argmax(patches_U, ignore_list):
    """Row-wise argmax that skips concept indices in ignore_list."""
    sorted_indices = np.argsort(-patches_U, axis=1)
    patches_C = np.zeros(patches_U.shape[0], dtype=int)
    for i in range(patches_U.shape[0]):
        for idx in sorted_indices[i]:
            if idx not in ignore_list:
                patches_C[i] = idx
                break
    return patches_C


def _reverse_preprocess(images, mean, std):
    """Undo ImageNet-style normalize for display (values in [0, 1])."""
    mean = np.array(mean)
    std = np.array(std)
    unnormalized = (images * std[None, None, :]) + mean[None, None, :]
    return np.clip(unnormalized, 0, 1)


def _save_concepts(crops, crops_u, reverse=True, start=0, nb_crops=20,
                   save=False, save_dir=None):
    """Save a grid of top-activating patch crops for each concept."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for c_id in range(crops_u.shape[1]):
        crops_u_np = crops_u.numpy() if hasattr(crops_u, "numpy") else crops_u
        sorted_indices = np.argsort(crops_u_np[:, c_id])[::-1]
        best_crops = np.array(crops)[sorted_indices[start:start + nb_crops]]

        rows = ceil(nb_crops / 10)
        plt.figure(figsize=(30, 4 * rows))
        for i in range(nb_crops):
            plt.subplot(rows, 10, i + 1)
            img = np.array(best_crops[i])
            if reverse:
                img = _reverse_preprocess(img, mean, std)
            plt.imshow(img)
            plt.axis("off")

        if save and save_dir is not None:
            plt.savefig(
                os.path.join(save_dir, f"concept_{c_id}.png"),
                bbox_inches="tight",
                dpi=300,
            )
        print("\n\n")
