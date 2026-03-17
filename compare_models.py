"""
Compare three approaches for 4-class whale classification.

Approaches:
  1. 4-class:        Single 4-class model (No Whale, Humpback, Orca, Beluga).
  2. Binary+3-class: Binary detector (Whale / No Whale) cascaded with a
                     3-class species classifier (Humpback, Orca, Beluga).
                     The 3-class model must be evaluated on the complete
                     test set (all 4 species) so that binary false positives
                     are classified by the 3-class model rather than
                     silently defaulting to No Whale.
  3. Binary+4-class: Binary detector cascaded with the 4-class model.
                     The 4-class stage can also reject false positives
                     from the binary detector.

Reads the ``test_split_with_predictions.csv`` files that are already
saved in each checkpoint subfolder, so no model loading or GPU is needed.

Usage:
    python compare_models.py

    # Custom prediction CSV paths
    python compare_models.py \\
        --pred_binary  checkpoints/binary/test_split_with_predictions.csv \\
        --pred_3class  checkpoints/3class/test_split_with_predictions.csv \\
        --pred_4class  checkpoints/4class/test_split_with_predictions.csv

    # Save results to CSV
    python compare_models.py --output comparison_results.csv
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

CLASS_NAMES = ["No Whale", "Humpback", "Orca", "Beluga"]
APPROACH_NAMES = ["4-Class", "Binary+3-Class", "Binary+4-Class"]


# ---------------------------------------------------------------------------
# Data loading & alignment
# ---------------------------------------------------------------------------
def load_and_merge(
    pred_binary: str,
    pred_3class: str,
    pred_4class: str,
) -> pd.DataFrame:
    """Load the three prediction CSVs and merge them on ``spec_name``.

    The 4-class CSV is used as the base (left joins) so that every
    sample — including No Whale — is preserved.

    The 3-class CSV must contain predictions for the complete test set
    (all 4 species), so that the cascade evaluation is fair: binary
    false positives are classified by the 3-class model rather than
    silently defaulting to No Whale.

    Returns a DataFrame with columns:
        label_4class      – ground-truth in the 4-class label space
        pred_binary       – binary prediction (0 = No Whale, 1 = Whale)
        pred_3class       – 3-class prediction (0 = Humpback, 1 = Orca, 2 = Beluga)
        pred_4class       – 4-class prediction (0-3)
    """
    df_bin = pd.read_csv(pred_binary)
    df_3c = pd.read_csv(pred_3class)
    df_4c = pd.read_csv(pred_4class)

    print(f"  Loaded binary  predictions: {len(df_bin):,} rows")
    print(f"  Loaded 3-class predictions: {len(df_3c):,} rows")
    print(f"  Loaded 4-class predictions: {len(df_4c):,} rows")

    merged = (
        df_4c[["spec_name", "label", "prediction"]]
        .rename(columns={"label": "label_4class", "prediction": "pred_4class"})
        .merge(
            df_bin[["spec_name", "prediction"]].rename(columns={"prediction": "pred_binary"}),
            on="spec_name",
            how="left",
        )
        .merge(
            df_3c[["spec_name", "prediction"]].rename(columns={"prediction": "pred_3class"}),
            on="spec_name",
            how="left",
        )
    )

    merged["pred_binary"] = merged["pred_binary"].fillna(0).astype(int)
    merged["pred_3class"] = merged["pred_3class"].fillna(-1).astype(int)

    n_missing_3c = (merged["pred_3class"] == -1).sum()
    if n_missing_3c:
        print(f"  ⚠  {n_missing_3c:,} samples have no 3-class prediction "
              f"(not in {pred_3class}). For a fair cascade comparison, "
              f"run the 3-class model on the complete test set.")

    print(f"  Merged samples: {len(merged):,}")
    return merged


# ---------------------------------------------------------------------------
# Prediction strategies
# ---------------------------------------------------------------------------
def cascade_binary_3class(df: pd.DataFrame) -> np.ndarray:
    """Binary gate → 3-class species classifier.

    Binary pred = 0  →  final = 0 (No Whale)
    Binary pred = 1  →  final = 3-class pred + 1
        (0=Humpback→1, 1=Orca→2, 2=Beluga→3)

    The 3-class model is expected to have predictions for the complete
    test set.  This way, binary false positives are actually classified
    by the 3-class model (which has no "No Whale" class), reflecting
    real deployment behaviour.
    """
    final = np.zeros(len(df), dtype=int)
    is_whale = df["pred_binary"].values == 1
    final[is_whale] = df.loc[is_whale, "pred_3class"].values + 1
    return final


def cascade_binary_4class(df: pd.DataFrame) -> np.ndarray:
    """Binary gate → 4-class model.

    Binary pred = 0  →  final = 0 (No Whale)
    Binary pred = 1  →  final = 4-class prediction (can also be 0)
    """
    final = np.zeros(len(df), dtype=int)
    is_whale = df["pred_binary"].values == 1
    final[is_whale] = df.loc[is_whale, "pred_4class"].values
    return final


# ---------------------------------------------------------------------------
# Metrics & display
# ---------------------------------------------------------------------------
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Per-class precision / recall / F1."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    return dict(precision=precision, recall=recall, f1=f1, support=support)


def build_comparison_df(all_metrics: dict) -> pd.DataFrame:
    """Build a single DataFrame with one row per class and columns for
    Precision / Recall / F1 × each approach, matching the layout:

        Class | Prec 4-Class | Prec B+3 | Prec B+4 | Rec 4-Class | ... | F1 ...
    """
    rows = []
    for i, cls in enumerate(CLASS_NAMES):
        row = {"Class": cls}
        for metric in ("precision", "recall", "f1"):
            for approach in APPROACH_NAMES:
                col = f"{metric.capitalize()} {approach}"
                row[col] = all_metrics[approach][metric][i]
        rows.append(row)
    return pd.DataFrame(rows)


def print_comparison(df: pd.DataFrame) -> None:
    """Pretty-print a single table matching the reference layout:

    ┌──────────┬─── Precision ──┬──── Recall ────┬────── F1 ──────┐
    │  Class   │ 4-Cl │ B+3  │ B+4  │ ...
    """
    metrics = ("Precision", "Recall", "F1")
    approaches = APPROACH_NAMES

    # Column widths
    cls_w = max(len(c) for c in CLASS_NAMES) + 2       # class col
    val_w = 10                                          # value cols
    grp_w = val_w * len(approaches) + len(approaches)   # group span

    # Shortened approach labels for sub-header
    short = {"4-Class": "4-class", "Binary+3-Class": "Binary +\n3-class",
             "Binary+4-Class": "Binary +\n4-class"}

    # Top border
    sep_line = "+" + "-" * cls_w
    for _ in metrics:
        sep_line += "+" + ("-" * val_w + "+") * len(approaches)
    print(sep_line.rstrip("+") + "+")

    # Metric group header
    header1 = "|" + " " * cls_w
    for m in metrics:
        span = val_w * len(approaches) + (len(approaches) - 1)
        header1 += "|" + m.center(span)
    header1 += "|"
    print(header1)

    # Sub-header (approach names) — handle two-line labels
    line1_parts = ["|" + "Class".center(cls_w)]
    line2_parts = ["|" + " " * cls_w]
    for _ in metrics:
        for a in approaches:
            lines = short[a].split("\n")
            line1_parts.append(lines[0].center(val_w))
            line2_parts.append((lines[1] if len(lines) > 1 else "").center(val_w))
    sub1 = "|".join(line1_parts) + "|"
    sub2 = "|".join(line2_parts) + "|"
    print(sep_line.rstrip("+") + "+")
    print(sub1)
    print(sub2)
    print(sep_line.rstrip("+") + "+")

    # Data rows
    for _, row in df.iterrows():
        parts = ["|" + row["Class"].center(cls_w)]
        for m in metrics:
            for a in approaches:
                col = f"{m} {a}"
                parts.append(f"{row[col]:.4f}".center(val_w))
        print("|".join(parts) + "|")
        print(sep_line.rstrip("+") + "+")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare 4-Class / Binary+3-Class / Binary+4-Class "
                    "approaches using saved prediction CSVs.",
    )
    parser.add_argument(
        "--pred_binary",
        type=str,
        default="checkpoints/binary/test_split_with_predictions.csv",
        help="Binary model predictions CSV.",
    )
    parser.add_argument(
        "--pred_3class",
        type=str,
        default="checkpoints/3class/test_split_with_predictions.csv",
        help="3-class model predictions CSV.",
    )
    parser.add_argument(
        "--pred_4class",
        type=str,
        default="checkpoints/4class/test_split_with_predictions.csv",
        help="4-class model predictions CSV.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the comparison table as CSV.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Load & merge prediction CSVs
    # ------------------------------------------------------------------
    print("Loading prediction CSVs …")
    merged = load_and_merge(args.pred_binary, args.pred_3class, args.pred_4class)

    labels = merged["label_4class"].values

    # ------------------------------------------------------------------
    # Build per-approach predictions in the 4-class label space
    # ------------------------------------------------------------------
    all_preds = {
        "4-Class": merged["pred_4class"].values,
        "Binary+3-Class": cascade_binary_3class(merged),
        "Binary+4-Class": cascade_binary_4class(merged),
    }

    # ------------------------------------------------------------------
    # Compute & display metrics
    # ------------------------------------------------------------------
    all_metrics = {
        name: compute_metrics(labels, preds) for name, preds in all_preds.items()
    }

    comparison_df = build_comparison_df(all_metrics)
    print_comparison(comparison_df)

    # ------------------------------------------------------------------
    # Optionally save to CSV
    # ------------------------------------------------------------------
    if args.output:
        comparison_df.to_csv(args.output, index=False, float_format="%.4f")
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
