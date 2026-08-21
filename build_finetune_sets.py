"""
build_finetune_sets.py — turn a manually-verified inference CSV into the
train / val / test split files used for fine-tuning (active learning).

This is the glue step between "I reviewed the model's detections" and
"I fine-tune the model on my corrections". It does three things:

  1. DIAGNOSE  — reads your verified CSV and reports where the model is failing
                 (false positives / missed whales / species confusion), then
                 suggests which strategy preset fits.
  2. ASSEMBLE  — selects and labels rows according to a small *strategy* YAML,
                 so the same tool covers very different situations (hard
                 negatives for a noisy site, species correction for a site
                 where detection is fine, etc.) without a fixed recipe.
  3. SPLIT     — writes group-aware train/val/test CSVs in the exact format
                 train.py expects (columns `spec_name`, `label`).

Label scheme (matches the inference cascade `pred_label` and configs/*.yaml):
    0 = No Whale   1 = Humpback   2 = Orca   3 = Beluga

Your verified CSV must contain a `verified_label` column using that same
0/1/2/3 scheme (i.e. you correct the model's `pred_label`). Everything else in
the CSV is ignored, so its exact contents can differ from site to site.

Examples
--------
    # Just look at the failure modes and get a suggested strategy:
    python build_finetune_sets.py --verified_csv verified.csv --diagnose_only

    # Build the split files using a preset strategy:
    python build_finetune_sets.py \
        --verified_csv verified.csv \
        --strategy hard_negatives \
        --output_dir data/cookinlet_splits
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import GroupShuffleSplit

# --------------------------------------------------------------------------- #
# Label scheme
# --------------------------------------------------------------------------- #
CLASS_NAMES = {0: "No Whale", 1: "Humpback", 2: "Orca", 3: "Beluga"}
SPECIES = {1: "Humpback", 2: "Orca", 3: "Beluga"}

# Default diagnostic thresholds (overridable per-strategy under `thresholds:`).
DEFAULT_THRESHOLDS = {
    "precision_min": 0.80,   # below this (with enough FP) => false-positive problem
    "recall_min": 0.80,      # below this (with enough FN) => missing-whale problem
    "binary_ok": 0.85,       # detection considered "fine" above this
    "species_metric_min": 0.80,
    "min_count": 30,         # ignore a signal backed by fewer than this many events
}

STRATEGY_DIR = os.path.join("configs", "active_learning")


# --------------------------------------------------------------------------- #
# Column resolution
# --------------------------------------------------------------------------- #
def resolve_spec_col(df: pd.DataFrame, requested: str) -> str:
    """Find the column holding the .npy spectrogram path."""
    if requested and requested != "auto":
        if requested not in df.columns:
            sys.exit(f"ERROR: --spec_col '{requested}' not found in CSV.")
        return requested
    for cand in ("spec_name", "file_path"):
        if cand in df.columns:
            return cand
    sys.exit(
        "ERROR: could not find a spectrogram-path column. Expected 'spec_name' "
        "or 'file_path'. Pass --spec_col explicitly."
    )


def pred_binary(df: pd.DataFrame) -> pd.Series:
    """Model's whale/no-whale decision as 0/1. Needs a prediction column."""
    if "pred_label_binary" in df.columns:
        return (df["pred_label_binary"].astype(int) != 0).astype(int)
    if "pred_label" in df.columns:
        return (df["pred_label"].astype(int) != 0).astype(int)
    sys.exit(
        "ERROR: this strategy/diagnostic needs the model's predictions, but the "
        "CSV has neither 'pred_label_binary' nor 'pred_label'. Use the inference "
        "output CSV (before you deleted the prediction columns)."
    )


def pred_species(df: pd.DataFrame) -> pd.Series:
    """Model's species decision as 1/2/3. Needs 'pred_label_3class'."""
    if "pred_label_3class" in df.columns:
        return df["pred_label_3class"].astype(int)
    if "pred_label" in df.columns:
        return df["pred_label"].astype(int)
    sys.exit(
        "ERROR: this strategy/diagnostic needs species predictions "
        "('pred_label_3class' or 'pred_label') and the CSV has neither."
    )


# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #
def _safe_div(a: int, b: int) -> float:
    return float(a) / float(b) if b else 0.0


def diagnose(df: pd.DataFrame, verified_col: str, th: dict) -> dict:
    """Compute binary + per-species precision/recall and suggest a strategy."""
    v = df[verified_col].astype(int)
    pb = pred_binary(df)

    v_whale = v.isin([1, 2, 3])
    p_whale = pb == 1

    tp = int((p_whale & v_whale).sum())
    fp = int((p_whale & ~v_whale).sum())
    fn = int((~p_whale & v_whale).sum())
    tn = int((~p_whale & ~v_whale).sum())

    bin_prec = _safe_div(tp, tp + fp)
    bin_rec = _safe_div(tp, tp + fn)

    print("\n" + "=" * 68)
    print("DIAGNOSTIC — where is the model failing on this batch?")
    print("=" * 68)
    print("\nDetection (whale vs. no-whale):")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision = {bin_prec:.3f}   recall = {bin_rec:.3f}")

    # Per-species metrics, computed on truly-whale windows only.
    species_metrics = {}
    whale_df = df[v_whale]
    if len(whale_df):
        ps = pred_species(whale_df)
        vs = whale_df[verified_col].astype(int)
        print("\nSpecies (on verified-whale windows only):")
        for cls, name in SPECIES.items():
            s_tp = int(((vs == cls) & (ps == cls)).sum())
            s_fp = int(((vs != cls) & (ps == cls)).sum())
            s_fn = int(((vs == cls) & (ps != cls)).sum())
            s_prec = _safe_div(s_tp, s_tp + s_fp)
            s_rec = _safe_div(s_tp, s_tp + s_fn)
            support = int((vs == cls).sum())
            species_metrics[cls] = dict(
                precision=s_prec, recall=s_rec, support=support, fn=s_fn, fp=s_fp
            )
            print(
                f"  {name:<9} support={support:<5} "
                f"precision={s_prec:.3f}  recall={s_rec:.3f}  "
                f"(confused: FN={s_fn}, FP={s_fp})"
            )
    else:
        print("\nSpecies: no verified-whale windows in this batch.")

    # --- Suggest a strategy from the numbers (informational) --------------- #
    suggestion, reason = _suggest_strategy(
        bin_prec, bin_rec, fp, fn, species_metrics, th
    )
    print("\n" + "-" * 68)
    print(f"SUGGESTED STRATEGY:  {suggestion}")
    print(f"  reason: {reason}")
    print(
        "  (This is a hint from the numbers — you decide. Override with "
        "--strategy.)"
    )
    print("-" * 68)

    return dict(
        precision=bin_prec, recall=bin_rec, tp=tp, fp=fp, fn=fn, tn=tn,
        species=species_metrics, suggestion=suggestion,
    )


def _suggest_strategy(prec, rec, fp, fn, species_metrics, th) -> tuple:
    mc = th["min_count"]
    if prec < th["precision_min"] and fp >= mc:
        return ("hard_negatives",
                f"low detection precision ({prec:.2f}) with FP={fp} "
                f">= {mc}: noise is triggering false positives.")
    if rec < th["recall_min"] and fn >= mc:
        return ("add_positives",
                f"low detection recall ({rec:.2f}) with FN={fn} "
                f">= {mc}: real whales are being missed.")
    detection_ok = prec >= th["binary_ok"] and rec >= th["binary_ok"]
    if detection_ok and species_metrics:
        worst = None
        for cls, m in species_metrics.items():
            if m["support"] < mc:
                continue
            metric = min(m["precision"], m["recall"])
            if metric < th["species_metric_min"] and (
                worst is None or metric < worst[1]
            ):
                worst = (cls, metric)
        if worst is not None:
            return ("species_correction",
                    f"detection is fine (P={prec:.2f}, R={rec:.2f}) but "
                    f"{SPECIES[worst[0]]} classification is weak "
                    f"({worst[1]:.2f}).")
    return ("balanced_refresh",
            "no single dominant failure mode — refresh both models with a "
            "class-balanced sample.")


# --------------------------------------------------------------------------- #
# Row-selection predicates (fixed vocabulary — no eval, no query language)
# --------------------------------------------------------------------------- #
def build_predicate(name: str, df: pd.DataFrame, verified_col: str) -> pd.Series:
    v = df[verified_col].astype(int)
    v_whale = v.isin([1, 2, 3])
    if name == "all":
        return pd.Series(True, index=df.index)
    if name == "true_whale":
        return v_whale
    if name in ("true_no_whale", "true_noise"):
        return v == 0
    pb = pred_binary(df)
    if name in ("false_positive", "fp"):
        return (pb == 1) & (v == 0)
    if name in ("false_negative", "fn"):
        return (pb == 0) & v_whale
    if name in ("true_positive", "tp"):
        return (pb == 1) & v_whale
    if name in ("misclassified_species", "species_error"):
        ps = pred_species(df)
        return v_whale & (pb == 1) & (ps != v)
    if name in ("correct_species",):
        ps = pred_species(df)
        return v_whale & (ps == v)
    sys.exit(
        f"ERROR: unknown selector '{name}' in strategy. Valid selectors: "
        "all, true_whale, true_no_whale, false_positive, false_negative, "
        "true_positive, misclassified_species, correct_species."
    )


def resolve_label(spec, verified_value: int):
    """Turn a strategy `label:` value into the integer label for a row."""
    if isinstance(spec, int):
        return spec
    if spec == "from_binary":
        return 0 if verified_value == 0 else 1
    if spec == "from_species":
        if verified_value not in (1, 2, 3):
            return None  # not a whale window; skip for the 3-class task
        return verified_value - 1  # 1/2/3 -> 0/1/2
    sys.exit(
        f"ERROR: invalid label spec '{spec}'. Use an integer, 'from_binary', "
        "or 'from_species'."
    )


# --------------------------------------------------------------------------- #
# Assemble one task (binary or 3class)
# --------------------------------------------------------------------------- #
def assemble_task(df, task_cfg, verified_col, spec_col, group_col):
    """Return a DataFrame with columns [spec_name, label, __group__, __os__]."""
    pool = set(df.index)
    picked_rows = []
    for rule in task_cfg:
        selector = rule["select"]
        label_spec = rule.get("label", "from_binary")
        oversample = int(rule.get("oversample", 1))
        mask = build_predicate(selector, df, verified_col)
        idx = [i for i in df.index[mask] if i in pool]
        for i in idx:
            vval = int(df.at[i, verified_col])
            label = resolve_label(label_spec, vval)
            if label is None:
                continue  # e.g. from_species on a no-whale row
            picked_rows.append(
                {
                    "spec_name": df.at[i, spec_col],
                    "label": int(label),
                    "__group__": df.at[i, group_col] if group_col else i,
                    "__os__": max(1, oversample),
                    "__src__": i,
                }
            )
            pool.discard(i)
    return pd.DataFrame(picked_rows)


# --------------------------------------------------------------------------- #
# Group-aware train / val / test split
# --------------------------------------------------------------------------- #
def grouped_split(df, train, val, test, seed):
    """Split rows into train/val/test keeping each group wholly in one split."""
    if df.empty:
        return df, df, df
    groups = df["__group__"].values
    n_groups = len(set(groups))
    if n_groups < 3:
        print(
            f"  WARNING: only {n_groups} distinct group(s) "
            f"('{df['__group__'].iloc[0]}'-style) — falling back to a random "
            "row-level split. Results may leak between splits; gather data from "
            "more recordings if you can."
        )
        shuffled = df.sample(frac=1.0, random_state=seed)
        n = len(shuffled)
        n_tr = int(round(n * train))
        n_va = int(round(n * val))
        return (
            shuffled.iloc[:n_tr],
            shuffled.iloc[n_tr:n_tr + n_va],
            shuffled.iloc[n_tr + n_va:],
        )

    idx = np.arange(len(df))
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(val + test), random_state=seed)
    tr_i, tmp_i = next(gss1.split(idx, groups=groups))
    tmp_groups = groups[tmp_i]
    rel_test = test / (val + test) if (val + test) > 0 else 0.0
    if len(set(tmp_groups)) < 2 or rel_test in (0.0, 1.0):
        # not enough groups to split val/test cleanly — put remainder in val
        va_i, te_i = tmp_i, np.array([], dtype=int)
    else:
        gss2 = GroupShuffleSplit(n_splits=1, test_size=rel_test, random_state=seed)
        va_rel, te_rel = next(gss2.split(tmp_i, groups=tmp_groups))
        va_i, te_i = tmp_i[va_rel], tmp_i[te_rel]
    return df.iloc[tr_i], df.iloc[va_i], df.iloc[te_i]


def apply_oversampling(train_df: pd.DataFrame) -> pd.DataFrame:
    """Repeat train rows by their per-rule oversample factor (train only!)."""
    if train_df.empty:
        return train_df
    repeated = train_df.loc[train_df.index.repeat(train_df["__os__"])]
    return repeated


def write_split(df, path, passthrough):
    cols = ["spec_name", "label"] + [c for c in passthrough if c in df.columns]
    out = df[cols] if not df.empty else pd.DataFrame(columns=cols)
    out.to_csv(path, index=False)


def label_distribution(df: pd.DataFrame) -> str:
    if df.empty:
        return "empty"
    counts = df["label"].value_counts().sort_index()
    return ", ".join(f"{int(k)}:{int(v)}" for k, v in counts.items())


# --------------------------------------------------------------------------- #
# Strategy loading
# --------------------------------------------------------------------------- #
def load_strategy(name_or_path: str) -> dict:
    path = name_or_path
    if not (os.path.sep in name_or_path or name_or_path.endswith((".yaml", ".yml"))):
        path = os.path.join(STRATEGY_DIR, f"{name_or_path}.yaml")
    if not os.path.exists(path):
        sys.exit(
            f"ERROR: strategy '{name_or_path}' not found (looked for '{path}'). "
            f"Built-in presets live in {STRATEGY_DIR}/."
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    print(f"\nLoaded strategy: {cfg.get('name', path)}")
    if cfg.get("description"):
        print(f"  {cfg['description']}")
    return cfg


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="Build train/val/test split CSVs from a verified inference "
        "CSV for fine-tuning (active learning).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--verified_csv", required=True,
                   help="Inference CSV with a verified_label column you filled in.")
    p.add_argument("--strategy", default=None,
                   help="Preset name (e.g. hard_negatives) or path to a strategy "
                        ".yaml. Omit with --diagnose_only.")
    p.add_argument("--output_dir", default=None,
                   help="Where to write the split CSVs.")
    p.add_argument("--spec_col", default="auto",
                   help="Column with the .npy spectrogram path (default: auto).")
    p.add_argument("--verified_col", default="verified_label",
                   help="Column with your verified 0/1/2/3 labels.")
    p.add_argument("--diagnose_only", action="store_true",
                   help="Only print the diagnostic + suggested strategy; write "
                        "nothing.")
    args = p.parse_args()

    if not os.path.exists(args.verified_csv):
        sys.exit(f"ERROR: verified CSV not found: {args.verified_csv}")
    df = pd.read_csv(args.verified_csv)

    if args.verified_col not in df.columns:
        sys.exit(
            f"ERROR: verified-label column '{args.verified_col}' not in CSV. "
            "Add a column with your corrected 0/1/2/3 labels "
            "(0=No Whale, 1=Humpback, 2=Orca, 3=Beluga)."
        )
    bad = ~df[args.verified_col].astype("Int64").isin([0, 1, 2, 3])
    if bad.any():
        sys.exit(
            f"ERROR: {int(bad.sum())} row(s) have a {args.verified_col} outside "
            "0/1/2/3. Fix these before continuing."
        )

    spec_col = resolve_spec_col(df, args.spec_col)
    print(f"Read {len(df)} verified rows from {args.verified_csv}")
    print(f"Spectrogram-path column: '{spec_col}'")

    # Merge strategy thresholds (if any) over defaults for the diagnostic.
    strategy = load_strategy(args.strategy) if args.strategy else None
    th = dict(DEFAULT_THRESHOLDS)
    if strategy and strategy.get("thresholds"):
        th.update(strategy["thresholds"])

    diagnose(df, args.verified_col, th)

    if args.diagnose_only:
        return
    if strategy is None:
        sys.exit(
            "\nNo --strategy given. Re-run with --strategy <name> to build the "
            "split files (or keep --diagnose_only)."
        )

    # ---- Assemble + split each requested task ---------------------------- #
    split_cfg = strategy.get("split", {})
    train = float(split_cfg.get("train", 0.70))
    val = float(split_cfg.get("val", 0.15))
    test = float(split_cfg.get("test", 0.15))
    seed = int(split_cfg.get("seed", 42))
    group_col = split_cfg.get("group_by", "audio")
    if group_col and group_col not in df.columns:
        print(
            f"\nWARNING: group_by column '{group_col}' not in CSV — splits will "
            "group by row instead (weaker leakage protection)."
        )
        group_col = None
    passthrough = [c for c in ("audio", "start(s)", "end(s)") if c in df.columns]

    out_dir = args.output_dir or os.path.join("data", "finetune_splits")
    os.makedirs(out_dir, exist_ok=True)

    task_map = {"binary": "binary", "3class": "3class"}
    tasks = strategy.get("tasks", [])
    include = strategy.get("include", {})
    if not tasks:
        sys.exit("ERROR: strategy defines no `tasks:` (binary and/or 3class).")

    print("\n" + "=" * 68)
    print(f"ASSEMBLING split files → {out_dir}")
    print(f"  split ratios: train={train} val={val} test={test}  seed={seed}")
    print(f"  group_by: {group_col or '(row-level fallback)'}")
    print("=" * 68)

    for task in tasks:
        if task not in task_map:
            sys.exit(f"ERROR: unknown task '{task}'. Use 'binary' or '3class'.")
        rules = include.get(task)
        if not rules:
            sys.exit(f"ERROR: strategy has task '{task}' but no include rules for it.")

        assembled = assemble_task(df, rules, args.verified_col, spec_col, group_col)
        if assembled.empty:
            print(f"\n[{task}] no rows selected — skipping.")
            continue

        tr, va, te = grouped_split(assembled, train, val, test, seed)
        tr = apply_oversampling(tr)

        for split_name, part in (("train", tr), ("val", va), ("test", te)):
            path = os.path.join(out_dir, f"{split_name}_{task}.csv")
            write_split(part, path, passthrough)

        print(f"\n[{task}]")
        print(f"  train: {len(tr):>5} rows  (labels {label_distribution(tr)})"
              f"  [after oversampling]")
        print(f"  val:   {len(va):>5} rows  (labels {label_distribution(va)})")
        print(f"  test:  {len(te):>5} rows  (labels {label_distribution(te)})")

    print("\nDone. Point train.py --train_csv / --val_csv at the files above, "
          "and keep the test_*.csv for measuring improvement.")


if __name__ == "__main__":
    main()
