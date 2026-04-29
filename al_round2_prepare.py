"""Build Round-2 training manifests for Tuxedni. Beluga-focused redo (no supplement).

Writes:
    data/tuxedni_splits/round2/train_binary.csv
    data/tuxedni_splits/round2/train_3class.csv
    data/tuxedni_splits/round2/val_binary.csv
    data/tuxedni_splits/round2/val_3class.csv
    data/tuxedni_splits/round2/final_test.csv

All rows carry an absolute `file_path` so training uses `x_col: file_path`.

Filters applied to the verified AL file:
  - drop rows whose audio_num is in the in-air range (mic out of water)
  - drop rows with manual_verif == "off_effort" (redundant with above, belt+suspenders)
  - drop deployment-118 rows (only 1 audio in existing splits; we train 120-only)

Window-level conflicts with existing training:
  - duplicate (same window, labels agree)  → drop the new AL row
  - conflict  (same window, labels differ) → drop the existing row; trust new label
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent
VERIFIED_CSV = REPO / "inference" / "tuxedni_results_1stAL_round_rev.csv"
EXISTING_BIN = REPO / "data" / "tuxedni_splits" / "train_binary.csv"
EXISTING_3C = REPO / "data" / "tuxedni_splits" / "train_3class.csv"
EXISTING_4C = REPO / "data" / "tuxedni_splits" / "train_4class_rev.csv"
FINAL_TEST_SRC = REPO / "data" / "tuxedni_splits" / "final_test.csv"
VAL_BIN_SRC = REPO / "data" / "splits_exp1" / "splits_binary" / "val_split_binary.csv"
VAL_3C_SRC = REPO / "data" / "splits_exp1" / "splits_3class" / "val_split.csv"

TUXEDNI_SPEC_DIR = (REPO / "data" / "tuxedni_spectrograms").resolve()
MULTISITE_SPEC_DIR = (REPO / "data" / "mel_spectrograms_multiclass").resolve()
FINAL_TEST_SPEC_DIR = (REPO / "data" / "tuxedni_final_test_spectrograms").resolve()
OUT_DIR = REPO / "data" / "tuxedni_splits" / "round2"

IN_AIR_MIN = 436
IN_AIR_MAX = 38194
DEP118_SOUND_ID = 0
SOUND_ID_START = 146
SR = 24000
CANONICAL_TO_3CLASS = {1: 0, 2: 1, 3: 2}
FINAL_COLS = ["window_id", "sound_id", "start", "end", "label", "file_path", "spec_name"]


def _audio_to_existing_sid() -> dict:
    """Map 'DEP_NNNNNNNN.e' → existing sound_id via train_4class_rev.sound_path."""
    df = pd.read_csv(EXISTING_4C)
    df["dep"] = df["sound_path"].str.extract(r"/ID(\d+)/")[0]
    df["audio_name"] = df["sound_path"].str.extract(r"/(\d+)\.e\.wav$")[0]
    df["audio_key"] = df["dep"] + "_" + df["audio_name"] + ".e"
    return dict(df.drop_duplicates("audio_key")[["audio_key", "sound_id"]].values)


def load_and_filter_new_al() -> pd.DataFrame:
    ver = pd.read_csv(VERIFIED_CSV)
    ver["mv"] = ver["manual_verif"].fillna("").astype(str).str.strip().str.lower()
    ver["audio_num"] = ver["audio"].str.extract(r"120_(\d+)\.e")[0].astype(int)
    in_air = (ver["audio_num"] < IN_AIR_MIN) | (ver["audio_num"] > IN_AIR_MAX)
    off_eff = ver["mv"] == "off_effort"
    keep = ver.loc[~(in_air | off_eff)].copy()
    print(f"  verified={len(ver)}  dropped_in_air_or_off_effort={int((in_air | off_eff).sum())}  "
          f"kept={len(keep)}")

    # Canonical label: 0=NoWhale (noise), 3=Beluga (when annotator re-labeled),
    # or pred_label when manual_verif is blank (prediction kept).
    def _lab(row):
        mv = row["mv"]
        if mv == "noise":
            return 0
        if mv == "beluga":
            return 3
        if mv == "":
            return int(row["pred_label"])
        raise ValueError(f"unexpected manual_verif={mv!r}")

    keep["label"] = keep.apply(_lab, axis=1)

    # Assign new sound_ids (stable per audio, insertion order from 146+)
    audios_in_order = keep["audio"].drop_duplicates().tolist()
    audio_to_sid = {a: SOUND_ID_START + i for i, a in enumerate(audios_in_order)}
    keep["sound_id"] = keep["audio"].map(audio_to_sid)

    keep = keep.sort_values(["sound_id", "start(s)"]).reset_index(drop=True)
    keep["window_id"] = keep.groupby("sound_id").cumcount()
    keep["start"] = (keep["start(s)"] * SR).round().astype(int)
    keep["end"] = (keep["end(s)"] * SR).round().astype(int)
    keep["spec_name"] = keep["file_path"].apply(os.path.basename)
    return keep


def resolve_conflicts(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """Return (kept_new_rows, existing_drops) — duplicates drop the new row,
    conflicts drop the existing row and keep the new one (trust newer verif)."""
    audio_to_old_sid = _audio_to_existing_sid()
    old_4c = pd.read_csv(EXISTING_4C)
    old_label_by_key = {
        f"{int(r['sound_id'])}|{int(r['start'])}|{int(r['end'])}": int(r["label"])
        for _, r in old_4c.iterrows()
    }

    keep_idx, drop_idx, existing_drops = [], [], []
    for idx, r in new_rows.iterrows():
        old_sid = audio_to_old_sid.get(r["audio"])
        if old_sid is None:
            keep_idx.append(idx); continue
        key = f"{old_sid}|{int(r['start'])}|{int(r['end'])}"
        if key not in old_label_by_key:
            keep_idx.append(idx); continue
        existing_lab = old_label_by_key[key]
        eb = 1 if existing_lab in (1, 2, 3) else 0
        nb = 1 if int(r["label"]) in (1, 2, 3) else 0
        if eb == nb:
            drop_idx.append(idx)  # duplicate — drop new
        else:
            keep_idx.append(idx)  # conflict — trust new
            existing_drops.append({
                "sound_id": old_sid, "start": int(r["start"]), "end": int(r["end"]),
                "audio": r["audio"], "existing_4c_label": existing_lab,
                "new_label": int(r["label"]),
            })
    return new_rows.loc[keep_idx].copy(), existing_drops


def _remove_rows(df: pd.DataFrame, drops: list) -> pd.DataFrame:
    if not drops or df.empty:
        return df.copy()
    keys = {(d["sound_id"], d["start"], d["end"]) for d in drops}
    mask = df.apply(
        lambda r: (int(r["sound_id"]), int(r["start"]), int(r["end"])) in keys, axis=1
    )
    return df.loc[~mask].copy()


def _absolutize(df: pd.DataFrame, spec_dir: Path) -> pd.DataFrame:
    df = df.copy()
    df["file_path"] = df["spec_name"].apply(lambda n: str((spec_dir / n).resolve()))
    return df


def _verify(df: pd.DataFrame, tag: str, n: int = 30) -> None:
    s = df["file_path"].drop_duplicates().sample(n=min(n, len(df)), random_state=0)
    missing = [p for p in s if not os.path.exists(p)]
    print(f"  [{tag}] rows={len(df)}  sampled={len(s)}  missing={len(missing)}")
    if missing:
        print(f"    first missing: {missing[0]}")
        sys.exit(1)


def main() -> None:
    print("Building round-2 manifests (beluga-focused; no humpback supplement)...")
    new_rows = load_and_filter_new_al()
    new_rows, existing_drops = resolve_conflicts(new_rows)
    print(f"  window conflict resolution: existing-rows-to-drop={len(existing_drops)} "
          f"new-kept={len(new_rows)}")
    for d in existing_drops:
        print(f"    conflict  sound_id={d['sound_id']}  start={d['start']}  end={d['end']}  "
              f"existing_label={d['existing_4c_label']} → new_label={d['new_label']}")

    # ---- existing tuxedni, cleaned ----
    def _clean_existing(path: Path) -> pd.DataFrame:
        df = pd.read_csv(path)
        if "Unnamed: 0" in df.columns:
            df = df.drop(columns=["Unnamed: 0"])
        df = df[df["sound_id"] != DEP118_SOUND_ID]
        df = _remove_rows(df, existing_drops)
        return _absolutize(df, TUXEDNI_SPEC_DIR)

    old_bin = _clean_existing(EXISTING_BIN)
    old_3c = _clean_existing(EXISTING_3C)

    # ---- new rows for binary (label 0/1) ----
    new_bin = new_rows.copy()
    new_bin["label"] = (new_bin["label"] > 0).astype(int)
    new_bin = new_bin[FINAL_COLS]

    bin_df = pd.concat([old_bin[FINAL_COLS], new_bin[FINAL_COLS]], ignore_index=True)
    bin_df = bin_df.sort_values(["sound_id", "start"]).reset_index(drop=True)
    bin_df["window_id"] = bin_df.groupby("sound_id").cumcount()

    # ---- new rows for 3-class (whales only, map canonical 1/2/3 → 0/1/2) ----
    new_3c = new_rows[new_rows["label"].isin([1, 2, 3])].copy()
    new_3c["label"] = new_3c["label"].map(CANONICAL_TO_3CLASS)
    new_3c = new_3c[FINAL_COLS]

    c3_df = pd.concat([old_3c[FINAL_COLS], new_3c[FINAL_COLS]], ignore_index=True)
    c3_df = c3_df.sort_values(["sound_id", "start"]).reset_index(drop=True)
    c3_df["window_id"] = c3_df.groupby("sound_id").cumcount()

    # ---- val + final_test copies with absolute file_path ----
    val_bin = pd.read_csv(VAL_BIN_SRC)
    val_bin["file_path"] = val_bin["spec_name"].apply(
        lambda n: str((MULTISITE_SPEC_DIR / n).resolve()))
    val_3c = pd.read_csv(VAL_3C_SRC)
    val_3c["file_path"] = val_3c["spec_name"].apply(
        lambda n: str((MULTISITE_SPEC_DIR / n).resolve()))
    final_test = pd.read_csv(FINAL_TEST_SRC)
    final_test["file_path"] = final_test["spec_name"].apply(
        lambda n: str((FINAL_TEST_SPEC_DIR / n).resolve()))

    print("\nPath sanity:")
    _verify(bin_df, "train binary")
    _verify(c3_df, "train 3class")
    _verify(val_bin, "val binary")
    _verify(val_3c, "val 3class")
    _verify(final_test, "final test")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    bin_df.to_csv(OUT_DIR / "train_binary.csv", index=False)
    c3_df.to_csv(OUT_DIR / "train_3class.csv", index=False)
    val_bin.to_csv(OUT_DIR / "val_binary.csv", index=False)
    val_3c.to_csv(OUT_DIR / "val_3class.csv", index=False)
    final_test.to_csv(OUT_DIR / "final_test.csv", index=False)

    print("\nManifests written:")
    for p, df in [(OUT_DIR / "train_binary.csv", bin_df),
                  (OUT_DIR / "train_3class.csv", c3_df),
                  (OUT_DIR / "val_binary.csv", val_bin),
                  (OUT_DIR / "val_3class.csv", val_3c),
                  (OUT_DIR / "final_test.csv", final_test)]:
        print(f"  {p.relative_to(REPO)}  rows={len(df)}  "
              f"labels={df['label'].value_counts().sort_index().to_dict()}")


if __name__ == "__main__":
    main()
