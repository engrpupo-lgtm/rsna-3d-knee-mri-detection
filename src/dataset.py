import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import pydicom
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as TF


class DualPlaneDataset(Dataset):
    """Loads cached 5-channel (2.5D) Sagittal and Coronal MRI tensors."""

    def __init__(
        self,
        df: pd.DataFrame,
        sag_dir: str,
        cor_dir: str,
        target_cols: List[str],
        image_size: int = 256,
        is_train: bool = True,
    ) -> None:
        self.df = df.reset_index(drop=True)
        self.sag_dir = sag_dir
        self.cor_dir = cor_dir
        self.target_cols = target_cols
        self.image_size = image_size
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.df)

    def _apply_augmentations(
        self, sag: torch.Tensor, cor: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Synchronized affine transform without horizontal flip (preserves medial/lateral anatomy)
        if random.random() > 0.5:
            angle = random.uniform(-10.0, 10.0)
            max_shift = 0.05 * self.image_size
            translate = [random.uniform(-max_shift, max_shift), random.uniform(-max_shift, max_shift)]
            scale = random.uniform(0.95, 1.05)

            sag = TF.affine(sag, angle=angle, translate=translate, scale=scale, shear=0.0)
            cor = TF.affine(cor, angle=angle, translate=translate, scale=scale, shear=0.0)

        return sag, cor

    def _load_plane_tensor(self, folder: str, study_id: str) -> torch.Tensor:
        path = os.path.join(folder, f"{study_id}.pt")
        if os.path.exists(path):
            data = torch.load(path, map_location="cpu", weights_only=True)
            return data["image"] if isinstance(data, dict) and "image" in data else data
        return torch.zeros((5, self.image_size, self.image_size), dtype=torch.float32)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        row = self.df.iloc[idx]
        study_id = str(row["StudyInstanceUID"])

        sag_tensor = self._load_plane_tensor(self.sag_dir, study_id)
        cor_tensor = self._load_plane_tensor(self.cor_dir, study_id)

        if self.is_train:
            sag_tensor, cor_tensor = self._apply_augmentations(sag_tensor, cor_tensor)

        labels = torch.tensor(row[self.target_cols].values.astype(np.float32), dtype=torch.float32)
        is_gold = bool(row["is_gold"]) if "is_gold" in row else False

        return {
            "sag": sag_tensor,
            "cor": cor_tensor,
            "targets": labels,
            "is_gold": is_gold,
            "study_id": study_id,
        }


def extract_central_25d_volume(
    series_folder: str, num_slices_out: int = 5, image_size: int = 256
) -> torch.Tensor:
    """Extracts contiguous central DICOM slices sorted along anatomical slice position."""
    resize_fn = T.Resize((image_size, image_size), antialias=True)
    if not os.path.exists(series_folder):
        return torch.zeros((num_slices_out, image_size, image_size), dtype=torch.float32)

    dcm_files = [
        os.path.join(series_folder, f)
        for f in os.listdir(series_folder)
        if f.endswith(".dcm")
    ]
    if not dcm_files:
        return torch.zeros((num_slices_out, image_size, image_size), dtype=torch.float32)

    slices = []
    for f in dcm_files:
        try:
            dcm = pydicom.dcmread(f, stop_before_pixels=True)
            pos = (
                float(dcm.ImagePositionPatient[2])
                if hasattr(dcm, "ImagePositionPatient")
                else float(dcm.InstanceNumber)
            )
            slices.append((pos, f))
        except Exception:
            continue

    if not slices:
        return torch.zeros((num_slices_out, image_size, image_size), dtype=torch.float32)

    slices.sort(key=lambda x: x[0])
    total = len(slices)
    center = total // 2
    half = num_slices_out // 2
    indices = [max(0, min(total - 1, center + offset)) for offset in range(-half, half + 1)]

    arrays = []
    for idx in indices:
        try:
            dcm = pydicom.dcmread(slices[idx][1])
            arrays.append(dcm.pixel_array.astype(np.float32))
        except Exception:
            arrays.append(np.zeros((image_size, image_size), dtype=np.float32))

    vol = np.stack(arrays)
    lo, hi = vol.min(), vol.max()
    vol = (vol - lo) / (hi - lo) if hi > lo else np.zeros_like(vol)
    return resize_fn(torch.tensor(vol, dtype=torch.float32))
