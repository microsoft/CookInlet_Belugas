"""Phase-1 cascade evaluation on final_test.csv at t=0.5.

Runs each Phase-1 binary checkpoint on final_test, then pairs with
(R1 3-class, R2 3-class, averaged 3-class). Also tries geometric-mean
binary ensembles combining each Phase-1 binary with R1's binary (which
was the Phase-0 winner ingredient).

Outputs a ranked table by Beluga F1 at threshold=0.5.
"""
from __future__ import annotations

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
TEST_CSV = REPO / "data" / "tuxedni_splits" / "round2" / "final_test.csv"
R1_PRED = REPO / "inference" / "final_test_round1_predictions.csv"
R2_PRED = REPO / "inference" / "final_test_round2_predictions.csv"

BINARY_CKPTS = {
    "R1":  REPO / "checkpoints" / "tuxedni_binary-finetune" / "best.ckpt",
    "R2":  REPO / "checkpoints" / "tuxedni_binary-r2-finetune" / "best.ckpt",
    "E1.1": REPO / "checkpoints" / "tuxedni_binary-e11-finetune" / "best.ckpt",
    "E1.2": REPO / "checkpoints" / "tuxedni_binary-e12-finetune" / "best.ckpt",
    "E1.3": REPO / "checkpoints" / "tuxedni_binary-e13-finetune" / "best.ckpt",
    "E1.4": REPO / "checkpoints" / "tuxedni_binary-e14-finetune" / "best.ckpt",
}

CLASS_NAMES = ["No Whale", "Humpback", "Orca", "Beluga"]
THRESHOLD = 0.5
TARGET_SIZE = (224, 180)


def _load(ckpt_path: str, device: str):
    model = ResNetClassifier.load_from_checkpoint(ckpt_path, strict=False)
    model.eval()
    model.freeze()
    return model.to(device)


def run_binary(ckpt: Path, df: pd.DataFrame, device: str) -> np.ndarray:
    model = _load(str(ckpt), device)
    ds = BioacousticsInferenceDataset(dataframe=df, x_col="file_path",
                                      target_size=TARGET_SIZE, normalize=True)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=2)
    logits = []
    with torch.no_grad():
        for x, _ in dl:
            o = model(x.to(device)).squeeze(1)
            logits.append(o.cpu().numpy())
    logits = np.concatenate(logits)
    return 1 / (1 + np.exp(-logits))


def cascade(bin_prob: np.ndarray, c3_probs: np.ndarray, threshold: float) -> np.ndarray:
    bin_pred = (bin_prob > threshold).astype(int)
    c3_pred = c3_probs.argmax(axis=1)
    return np.where(bin_pred == 0, 0, c3_pred + 1)


def metrics(y, pred):
    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(4)), average=None, zero_division=0)
    cm = np.zeros((4, 4), dtype=int)
    for t, pp in zip(y, pred):
        cm[int(t), int(pp)] += 1
    return dict(P=p, R=r, F1=f1, sup=sup, cm=cm,
                acc=cm.trace()/cm.sum(), macro_f1=f1.mean())


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_csv(TEST_CSV).sort_values("spec_name").reset_index(drop=True)
    y = df["label"].values.astype(int)

    # Collect binary probs for all checkpoints (reuse saved for R1/R2)
    r1 = pd.read_csv(R1_PRED).sort_values("spec_name").reset_index(drop=True)
    r2 = pd.read_csv(R2_PRED).sort_values("spec_name").reset_index(drop=True)
    assert (df["spec_name"].values == r1["spec_name"].values).all()
    assert (df["spec_name"].values == r2["spec_name"].values).all()

    bin_probs = {
        "R1":  r1["prob_whale"].values,
        "R2":  r2["prob_whale"].values,
    }
    for tag in ["E1.1", "E1.2", "E1.3", "E1.4"]:
        print(f"Running binary {tag} on final_test...")
        bin_probs[tag] = run_binary(BINARY_CKPTS[tag], df, device)

    c3_r1 = r1[["prob_humpback", "prob_orca", "prob_beluga"]].values
    c3_r2 = r2[["prob_humpback", "prob_orca", "prob_beluga"]].values
    c3_avg = (c3_r1 + c3_r2) / 2
    c3_options = {"R1 3c": c3_r1, "R2 3c": c3_r2, "avg 3c": c3_avg}

    results = []

    # Single-binary × each 3-class
    for b_tag, bp in bin_probs.items():
        for c_tag, c3 in c3_options.items():
            m = metrics(y, cascade(bp, c3, THRESHOLD))
            results.append({
                "exp": f"{b_tag} × {c_tag}",
                "bel_P": m["P"][3], "bel_R": m["R"][3], "bel_F1": m["F1"][3],
                "nw_F1": m["F1"][0], "acc": m["acc"], "macro_F1": m["macro_f1"],
                "cm": m["cm"],
            })

    # Ensemble binary: geom-mean of R1 and each Phase-1 variant × each 3-class
    for b_tag in ["E1.1", "E1.2", "E1.3", "E1.4", "R2"]:
        pg = np.sqrt(bin_probs["R1"] * bin_probs[b_tag])
        for c_tag, c3 in c3_options.items():
            m = metrics(y, cascade(pg, c3, THRESHOLD))
            results.append({
                "exp": f"geom(R1,{b_tag}) × {c_tag}",
                "bel_P": m["P"][3], "bel_R": m["R"][3], "bel_F1": m["F1"][3],
                "nw_F1": m["F1"][0], "acc": m["acc"], "macro_F1": m["macro_f1"],
                "cm": m["cm"],
            })

    # Full ensemble: geom-mean of R1, R2, and each Phase-1 variant
    for b_tag in ["E1.1", "E1.2", "E1.3", "E1.4"]:
        pg = np.cbrt(bin_probs["R1"] * bin_probs["R2"] * bin_probs[b_tag])
        for c_tag, c3 in [("avg 3c", c3_avg)]:
            m = metrics(y, cascade(pg, c3, THRESHOLD))
            results.append({
                "exp": f"geom(R1,R2,{b_tag}) × {c_tag}",
                "bel_P": m["P"][3], "bel_R": m["R"][3], "bel_F1": m["F1"][3],
                "nw_F1": m["F1"][0], "acc": m["acc"], "macro_F1": m["macro_f1"],
                "cm": m["cm"],
            })

    out = pd.DataFrame(results)
    out_show = out.drop(columns=["cm"]).sort_values("bel_F1", ascending=False).reset_index(drop=True)
    print("\n=== Phase-1 + ensembles (sorted by Beluga F1 @ t=0.5) ===")
    print(out_show.head(25).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    r1_base = out[out["exp"] == "R1 × R1 3c"]["bel_F1"].iloc[0]
    phase0_winner = 0.7200  # geom(R1,R2) × avg 3c from Phase 0
    winners_over_r1 = out_show[out_show["bel_F1"] > r1_base]
    winners_over_p0 = out_show[out_show["bel_F1"] > phase0_winner]

    print(f"\nBaselines: R1 cascade = {r1_base:.4f}  |  Phase-0 winner (E0.3) = {phase0_winner:.4f}")
    print(f"\nBeating R1 ({r1_base:.4f}): {len(winners_over_r1)} variants")
    print(f"Beating Phase-0 winner ({phase0_winner:.4f}): {len(winners_over_p0)} variants")

    # top-3 confusion matrices
    print("\n=== Top 3 confusion matrices ===")
    for i in range(min(3, len(out_show))):
        row = out_show.iloc[i]
        cm = out[out["exp"] == row["exp"]]["cm"].iloc[0]
        print(f"\n#{i+1}: {row['exp']}  (bel_F1={row['bel_F1']:.4f})")
        print(f"{'':>10}" + "".join(f"{c:>10}" for c in CLASS_NAMES))
        for k, c in enumerate(CLASS_NAMES):
            print(f"{c:>10}" + "".join(f"{cm[k, j]:>10d}" for j in range(4)))

    out_show.to_csv(REPO / "inference" / "phase1_results.csv", index=False, float_format="%.4f")
    print("\nResults saved to: inference/phase1_results.csv")


if __name__ == "__main__":
    main()
