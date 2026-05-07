"""Calibrate E1.4 binary + R1 3-class by temperature scaling (NLL-minimising
T fitted on the round-2 val splits) and run the cascade on the remaining
dep-120 Tuxedni windows (in-air and training windows excluded).

Outputs:
  inference/tuxedni_r2_resting_predictions.csv
      Columns: audio, start(s), end(s), spec_name,
               prob_whale (calibrated), prob_humpback/orca/beluga (calibrated),
               pred_binary@0.5, pred_canonical@0.5
  logs/round2/calibration.json
      {"T_binary": ..., "T_3class": ..., "val_nll_before": ..., ...}

Design:
  - Temperature scaling PRESERVES decisions at t=0.5 (binary) and argmax (3c).
    Calibration here is for honest probability outputs (ranking, future AL,
    downstream users setting their own thresholds). F1@0.5 on final_test is
    expected to be identical before/after calibration.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import BioacousticsInferenceDataset
from PytorchWildlife.models.bioacoustics import ResNetClassifier

REPO = Path(__file__).resolve().parent

BINARY_CKPT = REPO / "checkpoints" / "tuxedni_binary-e14-finetune" / "best.ckpt"
THREECLASS_CKPT = REPO / "checkpoints" / "tuxedni_3class-finetune" / "best.ckpt"  # R1 3-class
VAL_BINARY = REPO / "data" / "tuxedni_splits" / "round2" / "val_binary.csv"
VAL_3CLASS = REPO / "data" / "tuxedni_splits" / "round2" / "val_3class.csv"
FINAL_TEST = REPO / "data" / "tuxedni_splits" / "round2" / "final_test.csv"
SRC_INFERENCE = REPO / "inference" / "tuxedni_results.csv"

TRAIN_BINARY = REPO / "data" / "tuxedni_splits" / "round2" / "train_binary.csv"
TRAIN_3CLASS = REPO / "data" / "tuxedni_splits" / "round2" / "train_3class.csv"

OUT_CSV = REPO / "inference" / "tuxedni_r2_resting_predictions.csv"
CAL_JSON = REPO / "logs" / "round2" / "calibration.json"

IN_AIR_MIN, IN_AIR_MAX = 436, 38194
TARGET_SIZE = (224, 180)
INFER_BATCH = 128
INFER_WORKERS = 0  # avoid /dev/shm exhaustion (container has 64 MB shm)


# --------------------------------------------------------------------
#  Model loading + inference
# --------------------------------------------------------------------
def _load(ckpt: str, device: str):
    m = ResNetClassifier.load_from_checkpoint(ckpt, strict=False)
    m.eval(); m.freeze()
    return m.to(device)


def run_binary_logits(ckpt: str, df: pd.DataFrame, device: str) -> np.ndarray:
    m = _load(ckpt, device)
    ds = BioacousticsInferenceDataset(
        dataframe=df, x_col="file_path", target_size=TARGET_SIZE, normalize=True)
    dl = DataLoader(ds, batch_size=INFER_BATCH, shuffle=False, num_workers=INFER_WORKERS)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            o = m(x.to(device)).squeeze(1)  # [B]
            out.append(o.cpu().numpy())
    return np.concatenate(out)


def run_3class_logits(ckpt: str, df: pd.DataFrame, device: str) -> np.ndarray:
    m = _load(ckpt, device)
    ds = BioacousticsInferenceDataset(
        dataframe=df, x_col="file_path", target_size=TARGET_SIZE, normalize=True)
    dl = DataLoader(ds, batch_size=INFER_BATCH, shuffle=False, num_workers=INFER_WORKERS)
    out = []
    with torch.no_grad():
        for x, _ in dl:
            o = m(x.to(device))  # [B, 3]
            out.append(o.cpu().numpy())
    return np.concatenate(out)


# --------------------------------------------------------------------
#  Temperature scaling
# --------------------------------------------------------------------
def fit_temperature_binary(logits: np.ndarray, labels: np.ndarray, n_steps: int = 200) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.float32)
    log_T = torch.zeros(1, requires_grad=True)  # T = exp(log_T), ensures T > 0
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=n_steps, line_search_fn="strong_wolfe")

    def nll():
        opt.zero_grad()
        T = log_T.exp()
        loss = F.binary_cross_entropy_with_logits(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(nll)
    return float(log_T.detach().exp().item())


def fit_temperature_multiclass(logits: np.ndarray, labels: np.ndarray, n_steps: int = 200) -> float:
    logits_t = torch.tensor(logits, dtype=torch.float32)
    labels_t = torch.tensor(labels, dtype=torch.long)
    log_T = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.1, max_iter=n_steps, line_search_fn="strong_wolfe")

    def nll():
        opt.zero_grad()
        T = log_T.exp()
        loss = F.cross_entropy(logits_t / T, labels_t)
        loss.backward()
        return loss

    opt.step(nll)
    return float(log_T.detach().exp().item())


def bce_nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    return float(F.binary_cross_entropy_with_logits(
        torch.tensor(logits) / T, torch.tensor(labels, dtype=torch.float32)).item())


def ce_nll(logits: np.ndarray, labels: np.ndarray, T: float) -> float:
    return float(F.cross_entropy(
        torch.tensor(logits) / T, torch.tensor(labels, dtype=torch.long)).item())


# --------------------------------------------------------------------
#  Pipeline
# --------------------------------------------------------------------
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # --- Step 1: calibrate binary on val_binary ---
    print("\n[1/4] Calibrating E1.4 binary on val_binary...")
    vbin = pd.read_csv(VAL_BINARY)
    bin_logits_val = run_binary_logits(str(BINARY_CKPT), vbin, device)
    bin_labels_val = vbin["label"].values.astype(int)
    T_bin = fit_temperature_binary(bin_logits_val, bin_labels_val)
    nll_before = bce_nll(bin_logits_val, bin_labels_val, 1.0)
    nll_after  = bce_nll(bin_logits_val, bin_labels_val, T_bin)
    print(f"  T_binary = {T_bin:.4f}   val NLL  before={nll_before:.4f}  after={nll_after:.4f}")

    # --- Step 2: calibrate 3-class on val_3class ---
    print("\n[2/4] Calibrating R1 3-class on val_3class...")
    v3c = pd.read_csv(VAL_3CLASS)
    c3_logits_val = run_3class_logits(str(THREECLASS_CKPT), v3c, device)
    c3_labels_val = v3c["label"].values.astype(int)
    T_3c = fit_temperature_multiclass(c3_logits_val, c3_labels_val)
    nll_before_3c = ce_nll(c3_logits_val, c3_labels_val, 1.0)
    nll_after_3c  = ce_nll(c3_logits_val, c3_labels_val, T_3c)
    print(f"  T_3class = {T_3c:.4f}   val NLL  before={nll_before_3c:.4f}  after={nll_after_3c:.4f}")

    CAL_JSON.parent.mkdir(parents=True, exist_ok=True)
    CAL_JSON.write_text(json.dumps({
        "T_binary": T_bin,
        "T_3class": T_3c,
        "binary_val_nll_before": nll_before, "binary_val_nll_after": nll_after,
        "threeclass_val_nll_before": nll_before_3c, "threeclass_val_nll_after": nll_after_3c,
        "binary_ckpt": str(BINARY_CKPT), "threeclass_ckpt": str(THREECLASS_CKPT),
    }, indent=2))
    print(f"  Calibration saved to {CAL_JSON}")

    # --- Step 3: sanity-check on final_test ---
    print("\n[3/4] Sanity-check: calibrated cascade on final_test @ t=0.5")
    ft = pd.read_csv(FINAL_TEST)
    bin_logits_ft = run_binary_logits(str(BINARY_CKPT), ft, device)
    c3_logits_ft  = run_3class_logits(str(THREECLASS_CKPT), ft, device)
    bin_prob_uncal = 1 / (1 + np.exp(-bin_logits_ft))
    bin_prob_cal   = 1 / (1 + np.exp(-bin_logits_ft / T_bin))
    c3_prob_uncal = F.softmax(torch.tensor(c3_logits_ft), dim=1).numpy()
    c3_prob_cal   = F.softmax(torch.tensor(c3_logits_ft) / T_3c, dim=1).numpy()
    y_ft = ft["label"].values.astype(int)

    def cascade_f1_bel(bp, c3p, t):
        pred = np.where((bp > t).astype(int) == 0, 0, c3p.argmax(axis=1) + 1)
        from sklearn.metrics import precision_recall_fscore_support
        p, r, f1, _ = precision_recall_fscore_support(y_ft, pred, labels=list(range(4)),
                                                      average=None, zero_division=0)
        return f1[3], p[3], r[3]

    f1u, pu, ru = cascade_f1_bel(bin_prob_uncal, c3_prob_uncal, 0.5)
    f1c, pc, rc = cascade_f1_bel(bin_prob_cal,   c3_prob_cal,   0.5)
    print(f"  Uncalibrated  @0.5: Beluga P={pu:.4f} R={ru:.4f} F1={f1u:.4f}")
    print(f"  Calibrated    @0.5: Beluga P={pc:.4f} R={rc:.4f} F1={f1c:.4f}")
    # expected identical

    # --- Step 4: inference on remaining dep-120 ---
    print("\n[4/4] Running cascade on remaining dep-120 windows (exclude in-air + training)")
    src = pd.read_csv(SRC_INFERENCE)
    src = src[src["audio"].str.startswith("120_")].copy()
    src["audio_num"] = src["audio"].str.extract(r"120_(\d+)\.e")[0].astype(int)
    in_air = (src["audio_num"] < IN_AIR_MIN) | (src["audio_num"] > IN_AIR_MAX)
    src = src.loc[~in_air].copy()
    src["spec_name"] = src["file_path"].apply(os.path.basename)

    train_specs = set(pd.read_csv(TRAIN_BINARY)["spec_name"]) | set(pd.read_csv(TRAIN_3CLASS)["spec_name"])
    src = src.loc[~src["spec_name"].isin(train_specs)].copy()
    print(f"  Resting windows: {len(src):,}  audios: {src['audio'].nunique():,}")

    # New E1.4 binary inference
    print("  Running E1.4 binary on resting...")
    resting_bin_logits = run_binary_logits(str(BINARY_CKPT), src, device)
    prob_whale = 1 / (1 + np.exp(-resting_bin_logits / T_bin))

    # Reuse existing R1 3-class probs from source CSV (recover logits, apply T)
    c3_probs_src = src[["prob_class_1", "prob_class_2", "prob_class_3"]].values
    c3_logits_proxy = np.log(np.clip(c3_probs_src, 1e-12, 1.0))  # logits up to additive constant
    c3_probs_cal = F.softmax(torch.tensor(c3_logits_proxy) / T_3c, dim=1).numpy()

    # Cascade decisions at t=0.5
    bin_pred = (prob_whale > 0.5).astype(int)
    c3_pred = c3_probs_cal.argmax(axis=1)  # 0=Hump, 1=Orca, 2=Beluga
    canonical = np.where(bin_pred == 0, 0, c3_pred + 1)

    out = src[["audio", "start(s)", "end(s)", "spec_name", "file_path"]].copy()
    out["prob_whale"] = prob_whale
    out["prob_humpback"] = c3_probs_cal[:, 0]
    out["prob_orca"]     = c3_probs_cal[:, 1]
    out["prob_beluga"]   = c3_probs_cal[:, 2]
    out["pred_binary@0.5"] = bin_pred
    out["pred_3class"]     = c3_pred         # 0..2
    out["pred_canonical@0.5"] = canonical    # 0..3

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"  Wrote {OUT_CSV}  ({len(out):,} rows)")

    print("\nPrediction distribution (pred_canonical @ t=0.5):")
    print(out["pred_canonical@0.5"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
