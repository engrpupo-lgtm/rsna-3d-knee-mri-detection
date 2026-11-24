import argparse
import os
from typing import List

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF

from src.config import TrainConfig
from src.dataset import extract_central_25d_volume
from src.model import DualBranchKneeNet


def load_fold_models(
    checkpoint_paths: List[str], num_classes: int, in_channels: int, device: torch.device
) -> List[DualBranchKneeNet]:
    models_list = []
    for ckpt in checkpoint_paths:
        assert os.path.exists(ckpt), f"Checkpoint not found: {ckpt}"
        model = DualBranchKneeNet(
            num_classes=num_classes, in_channels=in_channels, weights_path=None
        ).to(device)
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
        model.eval()
        models_list.append(model)
    return models_list


@torch.no_grad()
def predict_study_with_tta(
    models_list: List[DualBranchKneeNet],
    sag: torch.Tensor,
    cor: torch.Tensor,
    use_tta: bool = True,
) -> np.ndarray:
    """Runs multi-fold ensemble inference with anatomy-preserving scale TTA (no horizontal flip)."""
    views = [(sag, cor)]
    if use_tta:
        for scale in (0.96, 1.04):
            sag_s = TF.affine(sag, angle=0.0, translate=[0.0, 0.0], scale=scale, shear=0.0)
            cor_s = TF.affine(cor, angle=0.0, translate=[0.0, 0.0], scale=scale, shear=0.0)
            views.append((sag_s, cor_s))

    all_probs = []
    for model in models_list:
        for v_sag, v_cor in views:
            logits = model(v_sag, v_cor)
            all_probs.append(torch.sigmoid(logits).cpu().numpy()[0])

    return np.mean(all_probs, axis=0)


def run_inference(
    test_series_csv: str,
    test_dicom_dir: str,
    sample_sub_csv: str,
    checkpoint_paths: List[str],
    output_csv: str = "submission.csv",
    use_tta: bool = True,
) -> pd.DataFrame:
    cfg = TrainConfig()
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    df_test = pd.read_csv(test_series_csv)
    sample_sub = pd.read_csv(sample_sub_csv)
    unique_studies = df_test["StudyInstanceUID"].unique()

    models_list = load_fold_models(
        checkpoint_paths, len(cfg.target_cols), cfg.in_channels, device
    )
    print(f"Loaded {len(models_list)} checkpoint(s) on {device} | TTA: {use_tta}")

    records = []
    for study_id in unique_studies:
        study_rows = df_test[df_test["StudyInstanceUID"] == study_id]

        sag_rows = study_rows[study_rows["Anatomical_Plane"] == "Sagittal"]
        if len(sag_rows) > 0:
            sag_path = os.path.join(
                test_dicom_dir, str(study_id), str(sag_rows.iloc[0]["SeriesInstanceUID"])
            )
            sag_t = extract_central_25d_volume(sag_path, cfg.in_channels, cfg.image_size)
        else:
            sag_t = torch.zeros((cfg.in_channels, cfg.image_size, cfg.image_size), dtype=torch.float32)

        cor_rows = study_rows[study_rows["Anatomical_Plane"] == "Coronal"]
        if len(cor_rows) > 0:
            cor_path = os.path.join(
                test_dicom_dir, str(study_id), str(cor_rows.iloc[0]["SeriesInstanceUID"])
            )
            cor_t = extract_central_25d_volume(cor_path, cfg.in_channels, cfg.image_size)
        else:
            cor_t = torch.zeros((cfg.in_channels, cfg.image_size, cfg.image_size), dtype=torch.float32)

        probs = predict_study_with_tta(
            models_list,
            sag_t.unsqueeze(0).to(device),
            cor_t.unsqueeze(0).to(device),
            use_tta=use_tta,
        )

        rec = {"StudyInstanceUID": study_id}
        for i, col in enumerate(cfg.target_cols):
            rec[col] = float(probs[i])
        records.append(rec)

    sub_df = pd.DataFrame(records)[sample_sub.columns]
    assert sub_df.isnull().sum().sum() == 0, "NaN values detected in submission."
    sub_df.to_csv(output_csv, index=False)
    print(f"Saved {output_csv} with shape {sub_df.shape}")
    return sub_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Fold RSNA Knee MRI Inference")
    parser.add_argument("--test-series-csv", type=str, required=True)
    parser.add_argument("--test-dicom-dir", type=str, required=True)
    parser.add_argument("--sample-sub-csv", type=str, required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--output-csv", type=str, default="submission.csv")
    parser.add_argument("--no-tta", action="store_true")
    args = parser.parse_args()

    run_inference(
        test_series_csv=args.test_series_csv,
        test_dicom_dir=args.test_dicom_dir,
        sample_sub_csv=args.sample_sub_csv,
        checkpoint_paths=args.checkpoints,
        output_csv=args.output_csv,
        use_tta=not args.no_tta,
    )
