"""Phase-0 post-hoc experiments (no retraining) to beat R1 Beluga F1 @ t=0.5.

Uses saved R1 and R2 prediction CSVs from inference/final_test_round*_predictions.csv.
Evaluates multiple combinations at the fixed threshold t=0.5:
  E0.2 arithmetic-mean of binary probs; average 3-class
  E0.3 geometric-mean of binary probs; average 3-class
  E0.4 R1 binary + averaged 3-class
  E0.5 R2 binary + averaged 3-class
  E0.6 probabilistic-cascade (no threshold gate): final = p_whale * p_3class; argmax over (noWhale, 3×whale)
  E0.7 noisy-OR ensemble of binary: 1 - (1-p_R1)(1-p_R2)
  E0.8 max-rule ensemble of binary: max(p_R1, p_R2)
  E0.9 min-rule ensemble of binary: min(p_R1, p_R2) (conservative)

Reports Beluga P/R/F1, accuracy, macro-F1 for each.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

REPO = Path(__file__).resolve().parent
R1 = REPO / "inference" / "final_test_round1_predictions.csv"
R2 = REPO / "inference" / "final_test_round2_predictions.csv"

CLASS_NAMES = ["No Whale", "Humpback", "Orca", "Beluga"]
T = 0.5


def cascade(bin_prob: np.ndarray, c3_probs: np.ndarray, threshold: float) -> np.ndarray:
    bin_pred = (bin_prob > threshold).astype(int)
    c3_pred = c3_probs.argmax(axis=1)
    return np.where(bin_pred == 0, 0, c3_pred + 1)


def metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    p, r, f1, sup = precision_recall_fscore_support(
        y, pred, labels=list(range(4)), average=None, zero_division=0)
    cm = np.zeros((4, 4), dtype=int)
    for t, pp in zip(y, pred):
        cm[int(t), int(pp)] += 1
    acc = cm.trace() / cm.sum()
    return {"P": p, "R": r, "F1": f1, "sup": sup, "cm": cm, "acc": acc,
            "macro_f1": f1.mean()}


def row(tag: str, m: dict) -> dict:
    return {
        "exp": tag,
        "bel_P": m["P"][3], "bel_R": m["R"][3], "bel_F1": m["F1"][3],
        "nowhale_F1": m["F1"][0], "orca_F1": m["F1"][2],
        "acc": m["acc"], "macro_F1": m["macro_f1"],
        "cm": m["cm"],
    }


def main() -> None:
    r1 = pd.read_csv(R1).sort_values("spec_name").reset_index(drop=True)
    r2 = pd.read_csv(R2).sort_values("spec_name").reset_index(drop=True)
    assert (r1["spec_name"].values == r2["spec_name"].values).all(), "spec_name misalignment"

    y = r1["label"].values.astype(int)
    p_r1 = r1["prob_whale"].values
    p_r2 = r2["prob_whale"].values
    c3_r1 = r1[["prob_humpback", "prob_orca", "prob_beluga"]].values
    c3_r2 = r2[["prob_humpback", "prob_orca", "prob_beluga"]].values
    c3_avg = (c3_r1 + c3_r2) / 2

    results = []

    # baselines (sanity check)
    results.append(row("R1 baseline @0.5", metrics(y, cascade(p_r1, c3_r1, T))))
    results.append(row("R2 baseline @0.5", metrics(y, cascade(p_r2, c3_r2, T))))

    # E0.2: avg binary, avg 3-class
    p_arith = (p_r1 + p_r2) / 2
    results.append(row("E0.2  avg(binary) avg(3c)", metrics(y, cascade(p_arith, c3_avg, T))))

    # E0.3: geometric mean binary, avg 3-class
    p_geom = np.sqrt(p_r1 * p_r2)
    results.append(row("E0.3  geom(binary) avg(3c)", metrics(y, cascade(p_geom, c3_avg, T))))

    # E0.4: R1 binary + avg 3-class
    results.append(row("E0.4  R1 binary + avg(3c)", metrics(y, cascade(p_r1, c3_avg, T))))

    # E0.5: R2 binary + avg 3-class
    results.append(row("E0.5  R2 binary + avg(3c)", metrics(y, cascade(p_r2, c3_avg, T))))

    # E0.6: probabilistic cascade — no threshold gate
    # final_prob[class_k] for k in {0 noWhale, 1 hump, 2 orca, 3 beluga}:
    #   k=0: 1 - p_whale   (noWhale prob)
    #   k=1: p_whale * p_3c[hump]
    #   k=2: p_whale * p_3c[orca]
    #   k=3: p_whale * p_3c[bel]
    for tag, pw, c3 in [("E0.6a R1-based prob-cascade", p_r1, c3_r1),
                        ("E0.6b R2-based prob-cascade", p_r2, c3_r2),
                        ("E0.6c ensemble prob-cascade", p_arith, c3_avg)]:
        probs = np.zeros((len(y), 4))
        probs[:, 0] = 1 - pw
        probs[:, 1] = pw * c3[:, 0]
        probs[:, 2] = pw * c3[:, 1]
        probs[:, 3] = pw * c3[:, 2]
        pred = probs.argmax(axis=1)
        results.append(row(tag, metrics(y, pred)))

    # E0.7: noisy-OR on binary
    p_nor = 1 - (1 - p_r1) * (1 - p_r2)
    results.append(row("E0.7  noisy-OR(binary) avg(3c)", metrics(y, cascade(p_nor, c3_avg, T))))

    # E0.8: max-rule on binary
    p_max = np.maximum(p_r1, p_r2)
    results.append(row("E0.8  max(binary) avg(3c)", metrics(y, cascade(p_max, c3_avg, T))))

    # E0.9: min-rule on binary (conservative)
    p_min = np.minimum(p_r1, p_r2)
    results.append(row("E0.9  min(binary) avg(3c)", metrics(y, cascade(p_min, c3_avg, T))))

    # Pretty-print ranked by Beluga F1
    df = pd.DataFrame(results)
    df_show = df.drop(columns=["cm"]).sort_values("bel_F1", ascending=False).reset_index(drop=True)
    print("\n=== Phase-0 results (sorted by Beluga F1 @ t=0.5) ===")
    print(df_show.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Highlight anything above R1 baseline
    r1_bel = df[df["exp"] == "R1 baseline @0.5"]["bel_F1"].iloc[0]
    winners = df_show[df_show["bel_F1"] > r1_bel]
    if len(winners) > 0:
        print(f"\n*** {len(winners)} variant(s) exceed R1 baseline (Beluga F1 {r1_bel:.4f}) ***")
        for _, r in winners.iterrows():
            print(f"    {r['exp']}: Beluga F1 = {r['bel_F1']:.4f}  (+{r['bel_F1']-r1_bel:.4f})")
    else:
        print(f"\n*** None beat R1 baseline (Beluga F1 = {r1_bel:.4f}) ***")

    # Show confusion matrix for top variant
    top = df_show.iloc[0]
    top_cm = df[df["exp"] == top["exp"]]["cm"].iloc[0]
    print(f"\n=== Top variant: {top['exp']} ===")
    print(f"{'':>10}" + "".join(f"{c:>10}" for c in CLASS_NAMES))
    for i, c in enumerate(CLASS_NAMES):
        print(f"{c:>10}" + "".join(f"{top_cm[i, j]:>10d}" for j in range(4)))

    df_show.to_csv(REPO / "inference" / "phase0_results.csv", index=False, float_format="%.4f")
    print(f"\nResults saved to: inference/phase0_results.csv")


if __name__ == "__main__":
    main()
