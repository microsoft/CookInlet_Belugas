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
    # Cascade comparison (default)
    python compare_models.py

    # Custom prediction CSV paths
    python compare_models.py \\
        --pred_binary        checkpoints/binary/test_split_with_predictions.csv \\
        --pred_3class        checkpoints/3class/test_split_with_predictions.csv \\
        --pred_4class        checkpoints/4class/test_split_with_predictions.csv \\
        --pred_4class_2stage checkpoints/4class/test_split_with_predictions.csv

    # Compare experiments (same model, different runs)
    python compare_models.py \\
        --compare  checkpoints/binary/test_split_with_predictions_old.csv \\
                   checkpoints/binary/test_split_with_predictions_new.csv \\
        --names old new

    # Binary+3-Class only (no 4-class CSVs needed)
    python compare_models.py --binary_3class_only \\
        --pred_binary  checkpoints/binary/test_split_with_predictions.csv \\
        --pred_3class  checkpoints/3class/test_split_with_predictions.csv

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
    pred_4class_2stage: str,
) -> pd.DataFrame:
    """Load the four prediction CSVs and merge them on ``spec_name``.

    The 4-class CSV is used as the base (left joins) so that every
    sample — including No Whale — is preserved.

    The 3-class CSV must contain predictions for the complete test set
    (all 4 species), so that the cascade evaluation is fair: binary
    false positives are classified by the 3-class model rather than
    silently defaulting to No Whale.

    Returns a DataFrame with columns:
        label_4class       – ground-truth in the 4-class label space
        pred_binary        – binary prediction (0 = No Whale, 1 = Whale)
        pred_3class        – 3-class prediction (0 = Humpback, 1 = Orca, 2 = Beluga)
        pred_4class        – 4-class prediction (0-3)
        pred_4class_2stage – 4-class prediction used in the Binary+4-Class cascade
    """
    df_bin = pd.read_csv(pred_binary)
    df_3c = pd.read_csv(pred_3class)
    df_4c = pd.read_csv(pred_4class)
    df_4c_2s = pd.read_csv(pred_4class_2stage)

    print(f"  Loaded binary         predictions: {len(df_bin):,} rows")
    print(f"  Loaded 3-class        predictions: {len(df_3c):,} rows")
    print(f"  Loaded 4-class        predictions: {len(df_4c):,} rows")
    print(f"  Loaded 4-class 2stage predictions: {len(df_4c_2s):,} rows")

    merged = (
        df_4c[["spec_name", "label", "prediction"]]
        .rename(columns={"label": "label_4class", "prediction": "pred_4class"})
        .merge(
            df_bin[["spec_name", "prediction"]].rename(
                columns={"prediction": "pred_binary"}
            ),
            on="spec_name",
            how="left",
        )
        .merge(
            df_3c[["spec_name", "prediction"]].rename(
                columns={"prediction": "pred_3class"}
            ),
            on="spec_name",
            how="left",
        )
        .merge(
            df_4c_2s[["spec_name", "prediction"]].rename(
                columns={"prediction": "pred_4class_2stage"}
            ),
            on="spec_name",
            how="left",
        )
    )

    merged["pred_binary"] = merged["pred_binary"].fillna(0).astype(int)
    merged["pred_3class"] = merged["pred_3class"].fillna(-1).astype(int)
    merged["pred_4class_2stage"] = merged["pred_4class_2stage"].fillna(0).astype(int)

    n_missing_3c = (merged["pred_3class"] == -1).sum()
    if n_missing_3c:
        print(
            f"  ⚠  {n_missing_3c:,} samples have no 3-class prediction "
            f"(not in {pred_3class}). For a fair cascade comparison, "
            f"run the 3-class model on the complete test set."
        )

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
    final[is_whale] = df.loc[is_whale, "pred_4class_2stage"].values
    return final


# ---------------------------------------------------------------------------
# Metrics & display
# ---------------------------------------------------------------------------
def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    """Per-class precision / recall / F1 and confusion counts (one-vs-rest)."""
    precision, recall, f1, support = precision_recall_fscore_support(
        labels,
        preds,
        labels=list(range(len(CLASS_NAMES))),
        average=None,
        zero_division=0,
    )
    n = len(CLASS_NAMES)
    tp = np.zeros(n, dtype=int)
    fp = np.zeros(n, dtype=int)
    fn = np.zeros(n, dtype=int)
    tn = np.zeros(n, dtype=int)
    for i in range(n):
        tp[i] = int(((labels == i) & (preds == i)).sum())
        fp[i] = int(((labels != i) & (preds == i)).sum())
        fn[i] = int(((labels == i) & (preds != i)).sum())
        tn[i] = int(((labels != i) & (preds != i)).sum())
    return dict(
        precision=precision,
        recall=recall,
        f1=f1,
        support=support,
        tp=tp,
        fp=fp,
        fn=fn,
        tn=tn,
    )


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
    cls_w = max(len(c) for c in CLASS_NAMES) + 2  # class col
    val_w = 10  # value cols
    grp_w = val_w * len(approaches) + len(approaches)  # group span

    # Shortened approach labels for sub-header
    short = {
        "4-Class": "4-class",
        "Binary+3-Class": "Binary +\n3-class",
        "Binary+4-Class": "Binary +\n4-class",
    }

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
# Binary+3-Class only
# ---------------------------------------------------------------------------
def load_and_merge_b3c(pred_binary: str, pred_3class: str) -> pd.DataFrame:
    """Load binary and 3-class CSVs and produce a 4-class ground-truth column.

    Two label spaces are supported, detected from the unique values in the
    binary CSV's ``label`` column:

    - **Binary label space** (labels ⊆ {0, 1}):
        binary label = 0  →  label_4class = 0  (No Whale)
        binary label = 1  →  label_4class = 3-class **prediction** + 1
      The 3-class CSV must contain predictions for the complete test set so
      that binary false positives are still classified by the 3-class model.

    - **4-class label space** (labels include 2 or 3):
        label_4class = label_binary  (used as-is — already 4-class)
      In this mode the 3-class CSV's prediction is informational only.
    """
    df_bin = pd.read_csv(pred_binary)
    df_3c = pd.read_csv(pred_3class)

    print(f"  Loaded binary  predictions: {len(df_bin):,} rows")
    print(f"  Loaded 3-class predictions: {len(df_3c):,} rows")

    merged = (
        df_bin[["spec_name", "label", "prediction"]]
        .rename(columns={"label": "label_binary", "prediction": "pred_binary"})
        .merge(
            df_3c[["spec_name", "prediction"]].rename(
                columns={"prediction": "pred_3class"}
            ),
            on="spec_name",
            how="left",
        )
    )
    merged["pred_3class"] = merged["pred_3class"].fillna(-1).astype(int)

    unique_labels = set(merged["label_binary"].dropna().astype(int).unique())
    if unique_labels.issubset({0, 1}):
        # Binary label space → reconstruct species via 3-class prediction
        print(
            "  Detected binary label space; reconstructing 4-class via 3-class prediction."
        )
        merged["label_4class"] = merged["label_binary"]
        is_whale = merged["label_binary"] == 1
        merged.loc[is_whale, "label_4class"] = (
            merged.loc[is_whale, "pred_3class"].clip(lower=0) + 1
        )
    else:
        # Already 4-class
        print(
            f"  Detected 4-class label space (labels: {sorted(unique_labels)}); using labels as-is."
        )
        merged["label_4class"] = merged["label_binary"].astype(int)

    n_missing = (merged["pred_3class"] == -1).sum()
    if n_missing:
        print(
            f"  ⚠  {n_missing:,} samples have no 3-class prediction "
            f"(not in {pred_3class})."
        )

    print(f"  Merged samples: {len(merged):,}")
    return merged


def print_single_approach(metrics: dict, approach_name: str) -> None:
    """Pretty-print metrics for a single approach."""
    cls_w = max(len(c) for c in CLASS_NAMES) + 2
    cnt_w = 8  # width for integer count columns
    val_w = 10  # width for float metric columns
    n_counts = 4  # FN, FP, TP, TN
    n_metrics = 3  # Precision, Recall, F1
    sep_line = (
        "+"
        + "-" * cls_w
        + ("+" + "-" * cnt_w) * n_counts
        + ("+" + "-" * val_w) * n_metrics
        + "+"
    )

    print(f"\n{approach_name}")
    print(sep_line)
    print(
        "|"
        + "Class".center(cls_w)
        + "|"
        + "FN".center(cnt_w)
        + "|"
        + "FP".center(cnt_w)
        + "|"
        + "TP".center(cnt_w)
        + "|"
        + "TN".center(cnt_w)
        + "|"
        + "Precision".center(val_w)
        + "|"
        + "Recall".center(val_w)
        + "|"
        + "F1".center(val_w)
        + "|"
    )
    print(sep_line)
    for i, cls in enumerate(CLASS_NAMES):
        print(
            "|"
            + cls.center(cls_w)
            + "|"
            + str(metrics["fn"][i]).center(cnt_w)
            + "|"
            + str(metrics["fp"][i]).center(cnt_w)
            + "|"
            + str(metrics["tp"][i]).center(cnt_w)
            + "|"
            + str(metrics["tn"][i]).center(cnt_w)
            + "|"
            + f"{metrics['precision'][i]:.4f}".center(val_w)
            + "|"
            + f"{metrics['recall'][i]:.4f}".center(val_w)
            + "|"
            + f"{metrics['f1'][i]:.4f}".center(val_w)
            + "|"
        )
        print(sep_line)


# ---------------------------------------------------------------------------
# Experiment comparison (same model, different runs)
# ---------------------------------------------------------------------------
def compare_experiments(
    csv_paths: list[str], exp_names: list[str], output: str | None = None
) -> None:
    """Compare multiple prediction CSVs from the same model type.

    Each CSV must have ``label`` and ``prediction`` columns.  Class names
    are inferred from the union of label values across all CSVs.
    """
    dfs = []
    for path, name in zip(csv_paths, exp_names):
        df = pd.read_csv(path)
        print(f"  {name}: {len(df):,} rows  ({path})")
        dfs.append(df)

    all_labels = sorted(set().union(*(df["label"].unique() for df in dfs)))
    n_classes = len(all_labels)

    all_metrics = {}
    for df, name in zip(dfs, exp_names):
        p, r, f1, sup = precision_recall_fscore_support(
            df["label"].values,
            df["prediction"].values,
            labels=all_labels,
            average=None,
            zero_division=0,
        )
        all_metrics[name] = dict(precision=p, recall=r, f1=f1, support=sup)

    # Build results DataFrame
    rows = []
    for i, cls in enumerate(all_labels):
        row = {"Class": cls}
        for metric in ("precision", "recall", "f1", "support"):
            for name in exp_names:
                col = f"{metric.capitalize()} {name}"
                row[col] = all_metrics[name][metric][i]
        rows.append(row)
    result_df = pd.DataFrame(rows)

    # Pretty-print
    _print_experiment_table(result_df, exp_names, all_labels)

    if output:
        result_df.to_csv(output, index=False, float_format="%.4f")
        print(f"\nResults saved to {output}")


def _print_experiment_table(
    df: pd.DataFrame, exp_names: list[str], class_labels: list
) -> None:
    """Pretty-print experiment comparison table."""
    metrics = ("Precision", "Recall", "F1")
    cls_w = max(max(len(str(c)) for c in class_labels), 5) + 2
    val_w = max(max(len(n) for n in exp_names) + 2, 10)

    sep_line = "+" + "-" * cls_w
    for _ in metrics:
        sep_line += "+" + ("-" * val_w + "+") * len(exp_names)
    sep_line = sep_line.rstrip("+") + "+"

    print(sep_line)

    header = "|" + " " * cls_w
    for m in metrics:
        span = val_w * len(exp_names) + (len(exp_names) - 1)
        header += "|" + m.center(span)
    print(header + "|")

    print(sep_line)
    sub = "|" + "Class".center(cls_w)
    for _ in metrics:
        for name in exp_names:
            sub += "|" + name.center(val_w)
    print(sub + "|")
    print(sep_line)

    for _, row in df.iterrows():
        parts = ["|" + str(row["Class"]).center(cls_w)]
        for m in metrics:
            for name in exp_names:
                col = f"{m} {name}"
                parts.append(f"{row[col]:.4f}".center(val_w))
        print("|".join(parts) + "|")
        print(sep_line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Compare model approaches or experiments using saved "
        "prediction CSVs.",
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
        help="4-class model predictions CSV (used for the standalone 4-class column).",
    )
    parser.add_argument(
        "--pred_4class_2stage",
        type=str,
        default="checkpoints/4class/test_split_with_predictions.csv",
        help="4-class model predictions CSV used for the Binary+4-Class cascade.",
    )
    parser.add_argument(
        "--compare",
        type=str,
        nargs="+",
        default=None,
        help="Two or more prediction CSVs to compare as experiments "
        "(same model, different runs).  When provided, the cascade "
        "comparison is skipped.",
    )
    parser.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Display names for --compare CSVs (must match length).",
    )
    parser.add_argument(
        "--binary_3class_only",
        action="store_true",
        help="Evaluate only the Binary+3-Class cascade. "
        "Requires --pred_binary and --pred_3class; "
        "--pred_4class and --pred_4class_2stage are ignored.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to save the comparison table as CSV.",
    )
    args = parser.parse_args()

    if args.compare:
        # -- Experiment comparison mode --
        if args.names:
            if len(args.names) != len(args.compare):
                parser.error(
                    "--names must have the same number of entries as --compare"
                )
            exp_names = args.names
        else:
            import os

            exp_names = [os.path.basename(p) for p in args.compare]

        print("Comparing experiments …")
        compare_experiments(args.compare, exp_names, output=args.output)
    elif args.binary_3class_only:
        # -- Binary+3-Class only mode --
        print("Loading prediction CSVs …")
        merged = load_and_merge_b3c(args.pred_binary, args.pred_3class)
        labels = merged["label_4class"].values
        preds = cascade_binary_3class(merged)
        metrics = compute_metrics(labels, preds)
        print_single_approach(metrics, "Binary+3-Class")

        if args.output:
            rows = [
                {
                    "Class": cls,
                    "FN": metrics["fn"][i],
                    "FP": metrics["fp"][i],
                    "TP": metrics["tp"][i],
                    "TN": metrics["tn"][i],
                    "Precision": metrics["precision"][i],
                    "Recall": metrics["recall"][i],
                    "F1": metrics["f1"][i],
                    "Support": metrics["support"][i],
                }
                for i, cls in enumerate(CLASS_NAMES)
            ]
            pd.DataFrame(rows).to_csv(args.output, index=False, float_format="%.4f")
            print(f"\nResults saved to {args.output}")
    else:
        # -- Cascade comparison mode (original) --
        print("Loading prediction CSVs …")
        merged = load_and_merge(
            args.pred_binary,
            args.pred_3class,
            args.pred_4class,
            args.pred_4class_2stage,
        )

        labels = merged["label_4class"].values

        all_preds = {
            "4-Class": merged["pred_4class"].values,
            "Binary+3-Class": cascade_binary_3class(merged),
            "Binary+4-Class": cascade_binary_4class(merged),
        }

        all_metrics = {
            name: compute_metrics(labels, preds) for name, preds in all_preds.items()
        }

        comparison_df = build_comparison_df(all_metrics)
        print_comparison(comparison_df)

        if args.output:
            comparison_df.to_csv(args.output, index=False, float_format="%.4f")
            print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
