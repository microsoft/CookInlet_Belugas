"""Write round-2 training manifests for Tuxedni.

Uses the dry-run logic in al_round2_dryrun.py. Produces:
    data/tuxedni_splits/round2/train_binary.csv
    data/tuxedni_splits/round2/train_3class.csv

All rows have an absolute `file_path` so the DataModule's `root` becomes
irrelevant. Round-2 configs must use `x_col: file_path`.
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from al_round2_dryrun import (
    CANONICAL_TO_3CLASS,
    EXISTING_TRAIN_BINARY,
    EXISTING_TRAIN_3CLASS,
    HUMPBACK_SUPPLEMENT_COUNT,
    REPO,
    SOUND_ID_START,
    _existing_audio_to_sid,
    apply_al_filters,
    build_new_al_rows,
    existing_train_3class_without_dep118,
    existing_train_binary_without_dep118,
    load_verified,
    remove_conflicting_rows,
    resolve_window_conflicts,
    sample_humpback_supplement,
)

TUXEDNI_SPEC_DIR = (REPO / "data" / "tuxedni_spectrograms").resolve()
MULTISITE_SPEC_DIR = (REPO / "data" / "mel_spectrograms_multiclass").resolve()
FINAL_TEST_SPEC_DIR = (REPO / "data" / "tuxedni_final_test_spectrograms").resolve()
OUT_DIR = REPO / "data" / "tuxedni_splits" / "round2"

VAL_BINARY_SRC = REPO / "data" / "splits_exp1" / "splits_binary" / "val_split_binary.csv"
VAL_3CLASS_SRC = REPO / "data" / "splits_exp1" / "splits_3class" / "val_split.csv"
FINAL_TEST_SRC = REPO / "data" / "tuxedni_splits" / "final_test.csv"

FINAL_COLS = ["window_id", "sound_id", "start", "end", "label", "file_path", "spec_name"]


def _absolutize_existing(df: pd.DataFrame, spec_dir: Path) -> pd.DataFrame:
    """Replace stale file_path with an absolute path derived from spec_name."""
    df = df.copy()
    df["file_path"] = df["spec_name"].apply(lambda n: str((spec_dir / n).resolve()))
    return df


def _verify_paths(df: pd.DataFrame, label: str, sample: int = 30) -> None:
    paths = df["file_path"].drop_duplicates()
    chk = paths.sample(n=min(sample, len(paths)), random_state=0)
    missing = [p for p in chk if not os.path.exists(p)]
    print(f"  [{label}] unique file_paths={len(paths)}  sampled={len(chk)}  missing={len(missing)}")
    if missing:
        print(f"    first 3 missing: {missing[:3]}")


def main() -> None:
    print("Building round-2 manifests...")

    # ---- 1. build + conflict-resolve new AL rows
    ver = load_verified()
    fil = apply_al_filters(ver)
    new_rows = build_new_al_rows(fil)
    new_rows, dropped_dups, existing_to_drop = resolve_window_conflicts(new_rows)
    print(f"  verified={len(ver)}  after filter={len(fil)}  "
          f"kept after conflict={len(new_rows)}  "
          f"dropped_dups={len(dropped_dups)}  existing_to_drop={len(existing_to_drop)}")

    # ---- 2. clean existing splits (drop dep-118 + conflict rows)
    old_bin = existing_train_binary_without_dep118()
    old_bin = remove_conflicting_rows(old_bin, existing_to_drop)
    old_3c = existing_train_3class_without_dep118()
    old_3c = remove_conflicting_rows(old_3c, existing_to_drop)
    if "Unnamed: 0" in old_bin.columns:
        old_bin = old_bin.drop(columns=["Unnamed: 0"])
    if "Unnamed: 0" in old_3c.columns:
        old_3c = old_3c.drop(columns=["Unnamed: 0"])
    old_bin = _absolutize_existing(old_bin, TUXEDNI_SPEC_DIR)
    old_3c = _absolutize_existing(old_3c, TUXEDNI_SPEC_DIR)

    # ---- 3. binary train: old + new (binary label 0/1)
    new_bin = new_rows.copy()
    new_bin["label"] = (new_bin["label"] > 0).astype(int)
    new_bin = new_bin[FINAL_COLS]

    bin_df = pd.concat([old_bin[FINAL_COLS], new_bin[FINAL_COLS]], ignore_index=True)
    bin_df = bin_df.sort_values(["sound_id", "start"]).reset_index(drop=True)
    bin_df["window_id"] = bin_df.groupby("sound_id").cumcount()

    # ---- 4. 3-class train: old (whales only) + new whales + humpback supplement
    new_3c = new_rows[new_rows["label"].isin([1, 2, 3])].copy()
    new_3c["label"] = new_3c["label"].map(CANONICAL_TO_3CLASS)
    new_3c = new_3c[FINAL_COLS]

    sid_offset = int(new_rows["sound_id"].max()) + 1 if len(new_rows) else SOUND_ID_START
    hump = sample_humpback_supplement(HUMPBACK_SUPPLEMENT_COUNT, sid_offset)
    # Multi-site rows have window_id and spec_name already; build absolute file_path
    hump["file_path"] = hump["spec_name"].apply(
        lambda n: str((MULTISITE_SPEC_DIR / n).resolve())
    )
    hump_out = hump[["window_id", "sound_id", "start", "end", "label", "file_path", "spec_name"]]

    c3_df = pd.concat([old_3c[FINAL_COLS], new_3c[FINAL_COLS], hump_out[FINAL_COLS]], ignore_index=True)
    c3_df = c3_df.sort_values(["sound_id", "start"]).reset_index(drop=True)
    c3_df["window_id"] = c3_df.groupby("sound_id").cumcount()

    # ---- 5. val + final_test copies with absolute file_path (round-2 sibling dir)
    val_bin = pd.read_csv(VAL_BINARY_SRC).copy()
    val_bin["file_path"] = val_bin["spec_name"].apply(
        lambda n: str((MULTISITE_SPEC_DIR / n).resolve())
    )
    val_3c = pd.read_csv(VAL_3CLASS_SRC).copy()
    val_3c["file_path"] = val_3c["spec_name"].apply(
        lambda n: str((MULTISITE_SPEC_DIR / n).resolve())
    )
    final_test = pd.read_csv(FINAL_TEST_SRC).copy()
    final_test["file_path"] = final_test["spec_name"].apply(
        lambda n: str((FINAL_TEST_SPEC_DIR / n).resolve())
    )

    # ---- 6. path sanity
    print("\nPath sanity:")
    _verify_paths(bin_df, "train binary")
    _verify_paths(c3_df, "train 3class")
    _verify_paths(val_bin, "val binary")
    _verify_paths(val_3c, "val 3class")
    _verify_paths(final_test, "final test")

    # ---- 7. write
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_path = OUT_DIR / "train_binary.csv"
    c3_path = OUT_DIR / "train_3class.csv"
    val_bin_path = OUT_DIR / "val_binary.csv"
    val_3c_path = OUT_DIR / "val_3class.csv"
    ft_path = OUT_DIR / "final_test.csv"
    bin_df.to_csv(bin_path, index=False)
    c3_df.to_csv(c3_path, index=False)
    val_bin.to_csv(val_bin_path, index=False)
    val_3c.to_csv(val_3c_path, index=False)
    final_test.to_csv(ft_path, index=False)

    print("\nWritten manifests:")
    for p, df in [(bin_path, bin_df), (c3_path, c3_df),
                  (val_bin_path, val_bin), (val_3c_path, val_3c),
                  (ft_path, final_test)]:
        print(f"  {p.relative_to(REPO)}  rows={len(df)}  "
              f"labels={df['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
