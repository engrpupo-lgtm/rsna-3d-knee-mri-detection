import argparse
import os
import random
import time
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.dataset import DualPlaneDataset
from src.model import DualBranchKneeNet


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prepare_folds(
    train_csv: str, llm_csv: str, target_cols: List[str], n_splits: int = 5, seed: int = 42
) -> pd.DataFrame:
    df_train = pd.read_csv(train_csv)
    df_llm = pd.read_csv(llm_csv)[["StudyInstanceUID"] + target_cols].copy()

    df = df_train.merge(df_llm, on="StudyInstanceUID", how="left", suffixes=("", "_llm"))
    df["is_gold"] = df[target_cols].notnull().all(axis=1)

    for col in target_cols:
        llm_col = f"{col}_llm"
        if llm_col in df.columns:
            df[col] = df[col].combine_first(df[llm_col])
        df[col] = df[col].astype(np.float32)

    assert df[target_cols].isnull().sum().sum() == 0, "Null values remain in targets."

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    df["fold"] = -1

    gold_idx = df[df["is_gold"]].index.to_numpy()
    for fold, (_, val_i) in enumerate(kf.split(gold_idx)):
        df.loc[gold_idx[val_i], "fold"] = fold

    weak_idx = df[~df["is_gold"]].index.to_numpy()
    for fold, (_, val_i) in enumerate(kf.split(weak_idx)):
        df.loc[weak_idx[val_i], "fold"] = fold

    return df


def compute_weighted_bce(
    logits: torch.Tensor,
    targets: torch.Tensor,
    is_gold: torch.Tensor,
    gold_weight: float = 3.0,
) -> torch.Tensor:
    bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    sample_weights = torch.where(is_gold, gold_weight, 1.0).unsqueeze(1)
    return (bce * sample_weights).mean()


def evaluate_macro_auc(
    y_true: np.ndarray, y_prob: np.ndarray, target_cols: List[str], threshold: float = 0.5
) -> Tuple[float, List[float]]:
    y_true_bin = (y_true >= threshold).astype(np.int32)
    class_aucs = []

    for i in range(len(target_cols)):
        yt = y_true_bin[:, i]
        yp = y_prob[:, i]
        if len(np.unique(yt)) > 1:
            class_aucs.append(float(roc_auc_score(yt, yp)))
        else:
            class_aucs.append(float("nan"))

    valid = [a for a in class_aucs if not np.isnan(a)]
    macro_auc = float(np.mean(valid)) if valid else 0.5
    return macro_auc, class_aucs


def train_fold(
    df: pd.DataFrame,
    sag_dir: str,
    cor_dir: str,
    cfg: TrainConfig,
    weights_path: str | None = None,
    output_dir: str = "checkpoints",
) -> float:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    use_amp = device.type == "cuda"

    train_df = df[df["fold"] != cfg.fold].reset_index(drop=True)
    val_df = df[df["fold"] == cfg.fold].reset_index(drop=True)

    train_loader = DataLoader(
        DualPlaneDataset(train_df, sag_dir, cor_dir, cfg.target_cols, cfg.image_size, is_train=True),
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        DualPlaneDataset(val_df, sag_dir, cor_dir, cfg.target_cols, cfg.image_size, is_train=False),
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    model = DualBranchKneeNet(
        num_classes=len(cfg.target_cols),
        in_channels=cfg.in_channels,
        weights_path=weights_path,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.epochs, eta_min=cfg.eta_min
    )
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_auc = 0.0
    val_is_gold = val_df["is_gold"].to_numpy()
    ckpt_path = os.path.join(output_dir, f"rsna_v2_dual_branch_fold{cfg.fold}_best.pth")

    print(f"Training Fold {cfg.fold} | Train: {len(train_df)} | Val: {len(val_df)} | Device: {device}")
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for batch in train_loader:
            sag = batch["sag"].to(device, non_blocking=True)
            cor = batch["cor"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            is_gold = batch["is_gold"].to(device, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(sag, cor)
                loss = compute_weighted_bce(logits, targets, is_gold, cfg.gold_sample_weight)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * sag.size(0)

        scheduler.step()
        train_loss = running_loss / len(train_df)

        model.eval()
        val_loss = 0.0
        preds, trues = [], []
        with torch.no_grad():
            for batch in val_loader:
                sag = batch["sag"].to(device, non_blocking=True)
                cor = batch["cor"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                is_gold = batch["is_gold"].to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_amp):
                    logits = model(sag, cor)
                    loss = compute_weighted_bce(logits, targets, is_gold, cfg.gold_sample_weight)

                val_loss += loss.item() * sag.size(0)
                preds.append(torch.sigmoid(logits).cpu().numpy())
                trues.append(targets.cpu().numpy())

        val_loss /= len(val_df)
        y_pred = np.vstack(preds)
        y_true = np.vstack(trues)

        overall_auc, _ = evaluate_macro_auc(y_true, y_pred, cfg.target_cols)
        gold_auc, _ = evaluate_macro_auc(y_true[val_is_gold], y_pred[val_is_gold], cfg.target_cols)

        mark = ""
        if overall_auc > best_auc:
            best_auc = overall_auc
            torch.save(model.state_dict(), ckpt_path)
            mark = "[BEST SAVED]"

        print(
            f"Epoch {epoch:02d}/{cfg.epochs:02d} [{time.time() - t0:.0f}s] | "
            f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
            f"Overall AUC: {overall_auc:.4f} | Gold AUC: {gold_auc:.4f} {mark}"
        )

    return best_auc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train RSNA 2.5D Dual-Branch Knee MRI Model")
    parser.add_argument("--train-csv", type=str, required=True)
    parser.add_argument("--llm-csv", type=str, required=True)
    parser.add_argument("--sag-dir", type=str, required=True)
    parser.add_argument("--cor-dir", type=str, required=True)
    parser.add_argument("--weights-path", type=str, default=None)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--gold-weight", type=float, default=3.0)
    args = parser.parse_args()

    cfg = TrainConfig(fold=args.fold, epochs=args.epochs, gold_sample_weight=args.gold_weight)
    seed_everything(cfg.seed)
    df = prepare_folds(args.train_csv, args.llm_csv, cfg.target_cols, cfg.n_splits, cfg.seed)
    train_fold(df, args.sag_dir, args.cor_dir, cfg, weights_path=args.weights_path)
