"""Round-2 cascade eval on final_test.csv with threshold sweep.

Runs the binary → 3-class cascade on every row of a test CSV, produces
canonical 4-class predictions, and sweeps the binary decision threshold
to find the operating point that maximises Beluga F1 (primary metric).

Usage:
    python al_round2_eval.py
    python al_round2_eval.py --tag round1 \
        --binary_ckpt checkpoints/tuxedni_binary-finetune/best.ckpt \
        --threeclass_ckpt checkpoints/tuxedni_3class-finetune/best.ckpt
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader

from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import BioacousticsInferenceDataset
from PytorchWildlife.models.bioacoustics import ResNetClassifier

REPO = Path(__file__).resolve().parent
DEFAULT_TEST_CSV = REPO / "data" / "tuxedni_splits" / "round2" / "final_test.csv"
DEFAULT_BIN_CKPT = REPO / "checkpoints" / "tuxedni_binary-r2-finetune" / "best.ckpt"
DEFAULT_3C_CKPT = REPO / "checkpoints" / "tuxedni_3class-r2-finetune" / "best.ckpt"

CLASS_NAMES = ["No Whale", "Humpback", "Orca", "Beluga"]


def _load_model(ckpt_path: str, device: str):
    """strict=False to tolerate loss-buffer mismatches (e.g. criterion.pos_weight)."""
    model = ResNetClassifier.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.freeze()
    return model.to(device)


def run_model(ckpt_path: str, df: pd.DataFrame, target_size: tuple, device: str,
              batch_size: int = 64, num_classes: int = 2) -> np.ndarray:
    model = _load_model(ckpt_path, device)
    ds = BioacousticsInferenceDataset(dataframe=df, x_col="file_path",
                                      target_size=target_size, normalize=True)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)
    logits = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.to(device)
            o = model(x)
            if num_classes == 2:
                o = o.squeeze(1)
            logits.append(o.cpu().numpy())
    logits = np.concatenate(logits)
    if num_classes == 2:
        return 1 / (1 + np.exp(-logits))  # sigmoid → whale prob
    return F.softmax(torch.tensor(logits), dim=1).numpy()


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    p, r, f1, sup = precision_recall_fscore_support(
        labels, preds, labels=list(range(len(CLASS_NAMES))),
        average=None, zero_division=0)
    cm = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=int)
    for t, pp in zip(labels, preds):
        cm[int(t), int(pp)] += 1
    return {"precision": p, "recall": r, "f1": f1, "support": sup, "cm": cm}


def cascade(bin_prob: np.ndarray, c3_pred: np.ndarray, threshold: float) -> np.ndarray:
    bin_pred = (bin_prob > threshold).astype(int)
    return np.where(bin_pred == 0, 0, c3_pred + 1)


def print_summary(m: dict, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'Class':<10} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Support':>8}")
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:<10} {m['precision'][i]:>8.4f} {m['recall'][i]:>8.4f} "
              f"{m['f1'][i]:>8.4f} {m['support'][i]:>8d}")
    acc = m["cm"].trace() / m["cm"].sum() if m["cm"].sum() else 0
    print(f"\naccuracy={acc:.4f}  macro-F1={m['f1'].mean():.4f}")
    print("Confusion (rows=truth, cols=pred):")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in CLASS_NAMES))
    for i, cls in enumerate(CLASS_NAMES):
        print(f"{cls:>10}" + "".join(f"{m['cm'][i, j]:>10d}" for j in range(len(CLASS_NAMES))))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_csv", default=str(DEFAULT_TEST_CSV))
    ap.add_argument("--binary_ckpt", default=str(DEFAULT_BIN_CKPT))
    ap.add_argument("--threeclass_ckpt", default=str(DEFAULT_3C_CKPT))
    ap.add_argument("--target_size", type=int, nargs=2, default=[224, 180])
    ap.add_argument("--tag", default="round2")
    ap.add_argument("--sweep_min", type=float, default=0.30)
    ap.add_argument("--sweep_max", type=float, default=0.90)
    ap.add_argument("--sweep_step", type=float, default=0.02)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(args.test_csv)
    print(f"Loaded {len(df)} rows from {args.test_csv}")
    print(f"Label distribution: {df['label'].value_counts().sort_index().to_dict()}")
    print(f"Device: {device}")

    print(f"\nRunning binary model: {args.binary_ckpt}")
    bin_prob = run_model(args.binary_ckpt, df, tuple(args.target_size), device, num_classes=2)

    print(f"\nRunning 3-class model: {args.threeclass_ckpt}")
    c3_probs = run_model(args.threeclass_ckpt, df, tuple(args.target_size), device, num_classes=3)
    c3_pred = c3_probs.argmax(axis=1)

    y = df["label"].values.astype(int)
    bel_idx = CLASS_NAMES.index("Beluga")  # = 3

    # --- threshold sweep ---
    thresholds = np.arange(args.sweep_min, args.sweep_max + 1e-9, args.sweep_step)
    rows = []
    for t in thresholds:
        pred = cascade(bin_prob, c3_pred, t)
        m = compute_metrics(y, pred)
        rows.append({
            "threshold": round(float(t), 3),
            "beluga_prec": m["precision"][bel_idx],
            "beluga_rec":  m["recall"][bel_idx],
            "beluga_f1":   m["f1"][bel_idx],
            "orca_f1":     m["f1"][2],
            "nowhale_f1":  m["f1"][0],
            "accuracy":    m["cm"].trace() / m["cm"].sum(),
            "macro_f1":    m["f1"].mean(),
        })
    sweep = pd.DataFrame(rows)
    print("\n=== Threshold sweep (top rows by beluga_f1) ===")
    print(sweep.sort_values("beluga_f1", ascending=False).head(10).to_string(index=False,
          float_format=lambda x: f"{x:.4f}"))

    best = sweep.loc[sweep["beluga_f1"].idxmax()]
    print(f"\nBest threshold for Beluga F1: {best['threshold']:.3f}  "
          f"(F1={best['beluga_f1']:.4f}  P={best['beluga_prec']:.4f}  R={best['beluga_rec']:.4f})")

    default = sweep.loc[np.argmin(np.abs(sweep["threshold"] - 0.5))]
    print(f"At default 0.5: F1={default['beluga_f1']:.4f}  "
          f"P={default['beluga_prec']:.4f}  R={default['beluga_rec']:.4f}")

    # detailed summary at best threshold and at 0.5
    for t, tag_suffix in [(default['threshold'], '@0.5'), (best['threshold'], '@best')]:
        pred = cascade(bin_prob, c3_pred, float(t))
        m = compute_metrics(y, pred)
        print_summary(m, f"{args.tag} (threshold={t:.3f}) {tag_suffix}")

    # persist predictions at best threshold + sweep
    pred_best = cascade(bin_prob, c3_pred, float(best["threshold"]))
    out = df[["spec_name", "label"]].copy()
    out["prob_whale"] = bin_prob
    out["prob_humpback"] = c3_probs[:, 0]
    out["prob_orca"] = c3_probs[:, 1]
    out["prob_beluga"] = c3_probs[:, 2]
    out["pred_3class"] = c3_pred
    out["pred_binary@best"] = (bin_prob > float(best["threshold"])).astype(int)
    out["pred_canonical@best"] = pred_best
    out["pred_canonical@0.5"] = cascade(bin_prob, c3_pred, 0.5)
    pred_path = REPO / "inference" / f"final_test_{args.tag}_predictions.csv"
    sweep_path = REPO / "inference" / f"final_test_{args.tag}_threshold_sweep.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(pred_path, index=False)
    sweep.to_csv(sweep_path, index=False, float_format="%.4f")
    print(f"\nPredictions: {pred_path}")
    print(f"Sweep:       {sweep_path}")


if __name__ == "__main__":
    main()
