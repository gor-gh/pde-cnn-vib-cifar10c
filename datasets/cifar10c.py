
import os
from typing import Optional, Callable, Tuple, List

import numpy as np
from torch.utils.data import Dataset

class CIFAR10C(Dataset):
    """
    CIFAR-10-C dataset loader.

    Expected directory structure (download separately):
      <root>/CIFAR-10-C/
        labels.npy
        gaussian_noise.npy
        shot_noise.npy
        impulse_noise.npy
        defocus_blur.npy
        glass_blur.npy
        motion_blur.npy
        zoom_blur.npy
        snow.npy
        frost.npy
        fog.npy
        brightness.npy
        contrast.npy
        elastic_transform.npy
        pixelate.npy
        jpeg_compression.npy

    Each corruption .npy is (50000, 32, 32, 3) uint8 in HWC format.
    labels.npy is (50000,) int64.
    Severities are contiguous blocks of 10k images: 1..5.
    """
    def __init__(
        self,
        root: str = "data",
        corruption: str = "gaussian_noise",
        severity: int = 3,
        transform: Optional[Callable] = None,
        base_folder: str = "CIFAR-10-C",
    ):
        if severity < 1 or severity > 5:
            raise ValueError("severity must be in {1,2,3,4,5}")
        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        cdir = os.path.join(root, base_folder)
        data_path = os.path.join(cdir, f"{corruption}.npy")
        labels_path = os.path.join(cdir, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Missing {data_path}. Download CIFAR-10-C and place it under {cdir}."
            )
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"Missing {labels_path}.")

        data = np.load(data_path)      # (50000,32,32,3) uint8
        labels = np.load(labels_path)  # (50000,)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.data = data[start:end]
        self.labels = labels[start:end].astype(np.int64)

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        x = self.data[idx]   # HWC uint8
        y = int(self.labels[idx])
        if self.transform is not None:
            x = self.transform(x)
        return x, y
