"""Dry-run preview for 2nd active-learning round on Tuxedni.

Reports what the new train manifests would contain without writing any files.
Deterministic (fixed seed) — re-running gives identical results.

Usage:
    conda activate bioacustics
    python al_round2_dryrun.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------------ config ------------------------------
SEED = 42
REPO = Path(__file__).resolve().parent

VERIFIED_CSV = REPO / "inference" / "tuxedni_results_1stAL_round_rev.csv"
EXISTING_TRAIN_BINARY = REPO / "data" / "tuxedni_splits" / "train_binary.csv"
EXISTING_TRAIN_3CLASS = REPO / "data" / "tuxedni_splits" / "train_3class.csv"
EXISTING_TRAIN_4CLASS = REPO / "data" / "tuxedni_splits" / "train_4class_rev.csv"
FINAL_TEST_CSV = REPO / "data" / "tuxedni_splits" / "final_test.csv"

VAL_BINARY = REPO / "data" / "splits_exp1" / "splits_binary" / "val_split_binary.csv"
VAL_3CLASS = REPO / "data" / "splits_exp1" / "splits_3class" / "val_split.csv"
MULTISITE_TRAIN_3CLASS = REPO / "data" / "splits_exp1" / "splits_3class" / "train_split.csv"

NEW_SPECTROGRAMS_DIR = Path(
    "/home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/"
    "Tuxedni_channel_CI/tuxedni_spectrograms"
)

IN_AIR_MIN = 436    # audios 0..435 are before mic-in-water
IN_AIR_MAX = 38194  # audios 38195.. are after mic recovered
DEP118_SOUND_ID_IN_SPLITS = 0  # the single dep-118 audio in existing splits

HUMPBACK_SUPPLEMENT_COUNT = 60  # target orca count in the new 3-class train
SOUND_ID_START = 146  # first new sound_id (existing range 0..145)

# canonical labels used in inference.py (pred_label)
# 0 = No Whale, 1 = Humpback, 2 = Orca, 3 = Beluga
# 3-class training schema (unchanged): 0=Humpback, 1=Orca, 2=Beluga
CANONICAL_TO_3CLASS = {1: 0, 2: 1, 3: 2}
# ------------------------------------------------------------------


def hdr(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def load_verified() -> pd.DataFrame:
    df = pd.read_csv(VERIFIED_CSV)
    df["mv"] = df["manual_verif"].fillna("").astype(str).str.strip().str.lower()
    df["audio_num"] = df["audio"].str.extract(r"120_(\d+)\.e").astype(int)
    return df


def apply_al_filters(df: pd.DataFrame) -> pd.DataFrame:
    in_air = (df["audio_num"] < IN_AIR_MIN) | (df["audio_num"] > IN_AIR_MAX)
    off_eff = df["mv"] == "off_effort"
    return df.loc[~(in_air | off_eff)].copy()


def canonical_label(row) -> int:
    mv = row["mv"]
    if mv == "noise":
        return 0
    if mv == "beluga":
        return 3
    if mv == "":
        return int(row["pred_label"])
    raise ValueError(f"Unexpected manual_verif={mv!r}")


SR = 24000  # sample rate (matches data/data_config.yaml)


def build_new_al_rows(filtered: pd.DataFrame) -> pd.DataFrame:
    rows = filtered.copy()
    rows["label"] = rows.apply(canonical_label, axis=1)
    # assign sound_id: one id per unique audio, stable order (first appearance)
    audios_in_order = rows["audio"].drop_duplicates().tolist()
    audio_to_sid = {a: SOUND_ID_START + i for i, a in enumerate(audios_in_order)}
    rows["sound_id"] = rows["audio"].map(audio_to_sid)

    # window_id: running counter within each sound_id (matches existing convention)
    rows = rows.sort_values(["sound_id", "start(s)"]).reset_index(drop=True)
    rows["window_id"] = rows.groupby("sound_id").cumcount()

    # start/end in SAMPLES (existing splits use sample indices, not seconds)
    rows["start"] = (rows["start(s)"] * SR).round().astype(int)
    rows["end"] = (rows["end(s)"] * SR).round().astype(int)

    # spec_name (filename only); absolute file_path stays as-is from the verified CSV
    rows["spec_name"] = rows["file_path"].apply(lambda p: os.path.basename(p))
    return rows


def _existing_audio_to_sid() -> dict:
    """Map 'DEP_NNNNNNNN.e' (e.g. '120_00008961.e') → existing sound_id.

    Uses train_4class_rev.sound_path as the source of truth for audio identity.
    """
    df = pd.read_csv(EXISTING_TRAIN_4CLASS)
    df["dep"] = df["sound_path"].str.extract(r"/ID(\d+)/")[0]
    df["audio_name"] = df["sound_path"].str.extract(r"/(\d+)\.e\.wav$")[0]
    df["audio_key"] = df["dep"] + "_" + df["audio_name"] + ".e"
    return dict(df.drop_duplicates("audio_key")[["audio_key", "sound_id"]].values)


def resolve_window_conflicts(new_rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Detect (audio, start, end) matches between new AL rows and existing training.

    Rule:
      * labels AGREE (at the binary level) → drop the new AL row (duplicate).
      * labels DISAGREE → keep new AL row; return the existing (sound_id, start, end)
        tuples that must be removed from binary / 3class / 4class manifests.

    Returns
    -------
    kept_new_rows : DataFrame (new AL rows to keep)
    dropped_new_rows : DataFrame (new AL rows dropped as duplicates)
    existing_to_drop : list of dicts {sound_id, start, end, reason, existing_label_*}
    """
    audio_to_sid = _existing_audio_to_sid()

    old_4c = pd.read_csv(EXISTING_TRAIN_4CLASS)
    old_4c_key = old_4c.assign(key=old_4c["sound_id"].astype(str) + "|" +
                                   old_4c["start"].astype(str) + "|" +
                                   old_4c["end"].astype(str)).set_index("key")["label"].to_dict()

    keep_idx = []
    drop_idx = []
    existing_to_drop = []

    for idx, r in new_rows.iterrows():
        old_sid = audio_to_sid.get(r["audio"])
        if old_sid is None:
            keep_idx.append(idx)
            continue
        key = f"{old_sid}|{int(r['start'])}|{int(r['end'])}"
        if key not in old_4c_key:
            keep_idx.append(idx)
            continue
        existing_label = int(old_4c_key[key])
        # binary collapse
        existing_binary = 1 if existing_label in (1, 2, 3) else 0
        new_binary = 1 if int(r["label"]) in (1, 2, 3) else 0
        if existing_binary == new_binary:
            # duplicate — drop the new AL row (option A)
            drop_idx.append(idx)
        else:
            # conflict — drop existing rows across all splits (option B); keep new AL row
            keep_idx.append(idx)
            existing_to_drop.append({
                "sound_id": old_sid, "start": int(r["start"]), "end": int(r["end"]),
                "audio": r["audio"], "existing_4c_label": existing_label,
                "new_canonical_label": int(r["label"]),
            })

    kept = new_rows.loc[keep_idx].copy()
    dropped = new_rows.loc[drop_idx].copy()
    return kept, dropped, existing_to_drop


def remove_conflicting_rows(df: pd.DataFrame, drops: list) -> pd.DataFrame:
    """Remove rows matching (sound_id, start, end) from `drops` out of df."""
    if not drops or df.empty:
        return df.copy()
    keys = {(d["sound_id"], d["start"], d["end"]) for d in drops}
    mask = df.apply(lambda r: (int(r["sound_id"]), int(r["start"]), int(r["end"])) in keys, axis=1)
    return df.loc[~mask].copy()


def existing_train_binary_without_dep118() -> pd.DataFrame:
    df = pd.read_csv(EXISTING_TRAIN_BINARY)
    return df[df["sound_id"] != DEP118_SOUND_ID_IN_SPLITS].copy()


def existing_train_3class_without_dep118() -> pd.DataFrame:
    df = pd.read_csv(EXISTING_TRAIN_3CLASS)
    return df[df["sound_id"] != DEP118_SOUND_ID_IN_SPLITS].copy()


def sample_humpback_supplement(n: int, sound_id_offset: int) -> pd.DataFrame:
    """Sample n humpback windows from multi-site train, one per unique source audio.

    Multi-site sound_ids are in a separate namespace (per-dataset); real audio
    identity is in (location, sound_filename). Reassign sound_ids starting at
    sound_id_offset so the merged manifest has globally unique ids.
    """
    df = pd.read_csv(MULTISITE_TRAIN_3CLASS)
    hump = df[df["label"] == 0].copy()
    rng = np.random.default_rng(SEED)
    # the original sound_id + location pair identifies a unique audio
    hump["orig_audio_key"] = hump["location"].astype(str) + "|" + hump["sound_id"].astype(str)
    unique_audios = hump["orig_audio_key"].drop_duplicates().tolist()
    chosen = rng.choice(unique_audios, size=min(n, len(unique_audios)), replace=False)
    # one random window per chosen audio
    out = (hump[hump["orig_audio_key"].isin(chosen)]
           .groupby("orig_audio_key")
           .sample(n=1, random_state=SEED)
           .reset_index(drop=True))
    # remap sound_ids to global namespace
    orig_to_new = {k: sound_id_offset + i for i, k in enumerate(out["orig_audio_key"].tolist())}
    out["orig_sound_id"] = out["sound_id"]
    out["sound_id"] = out["orig_audio_key"].map(orig_to_new)
    return out


def path_exists_check(paths: pd.Series, sample_n: int = 20) -> dict:
    s = paths.drop_duplicates()
    n_total = len(s)
    n_sample = min(sample_n, n_total)
    sample = s.sample(n=n_sample, random_state=SEED).tolist()
    missing = [p for p in sample if not os.path.exists(p)]
    return {"total_unique": n_total, "sampled": n_sample, "missing_sample": missing}


def main() -> None:
    hdr("AL Round-2 dry run — NO FILES WILL BE WRITTEN")

    # 1. verified & filtered
    ver = load_verified()
    fil = apply_al_filters(ver)
    hdr("1. Verified file + filters")
    print(f"  source: {VERIFIED_CSV}")
    print(f"  rows before filter: {len(ver)}")
    print(f"  in-air rows (audio_num <{IN_AIR_MIN} or >{IN_AIR_MAX}): "
          f"{int(((ver['audio_num'] < IN_AIR_MIN) | (ver['audio_num'] > IN_AIR_MAX)).sum())}")
    print(f"  off_effort rows: {int((ver['mv']=='off_effort').sum())}")
    print(f"  rows after filter: {len(fil)}")
    print(f"  unique audios after filter: {fil['audio'].nunique()}")
    if len(fil) != 1042:
        print(f"  WARNING: expected 1042 usable rows, got {len(fil)}")

    # 2. assign labels + sound_ids
    new_rows = build_new_al_rows(fil)
    hdr("2. New AL rows — canonical labels & sound_id assignment")
    print(f"  new rows: {len(new_rows)}")
    print(f"  unique audios: {new_rows['audio'].nunique()}")
    print(f"  sound_id range: "
          f"{new_rows['sound_id'].min()}..{new_rows['sound_id'].max()}")
    print("  label distribution (canonical):")
    print(new_rows["label"].value_counts().sort_index().to_string(header=False))

    # 2b. resolve (audio, start, end) conflicts against existing splits
    new_rows, dropped_dups, existing_to_drop = resolve_window_conflicts(new_rows)
    hdr("2b. Window-level conflict resolution")
    print(f"  new AL rows dropped as DUPLICATES (labels agree — option A): {len(dropped_dups)}")
    for _, r in dropped_dups.iterrows():
        print(f"    - {r['audio']}  start={r['start']}  end={r['end']}  "
              f"new_canonical_label={int(r['label'])}")
    print(f"  existing training rows to DROP (labels disagree — option B): "
          f"{len(existing_to_drop)}")
    for d in existing_to_drop:
        print(f"    - {d['audio']} (sound_id={d['sound_id']}) "
              f"start={d['start']} end={d['end']}  "
              f"existing_4c_label={d['existing_4c_label']} → new={d['new_canonical_label']}")
    print(f"  new AL rows kept after conflict pass: {len(new_rows)}")

    # 3. build binary-train preview
    old_bin = existing_train_binary_without_dep118()
    old_bin = remove_conflicting_rows(old_bin, existing_to_drop)
    hdr("3. BINARY train preview (round2)")
    raw_bin_total = len(pd.read_csv(EXISTING_TRAIN_BINARY))
    print(f"  existing train_binary.csv rows: {raw_bin_total}")
    print(f"  rows removed (dep-118 sound_id={DEP118_SOUND_ID_IN_SPLITS}): "
          f"{raw_bin_total - len(existing_train_binary_without_dep118())}")
    print(f"  rows removed (conflict option B): "
          f"{len(existing_train_binary_without_dep118()) - len(old_bin)}")
    # convert new AL rows to binary (label 0 stays 0, 1/2/3 → 1)
    new_bin = new_rows.copy()
    new_bin["label_binary"] = (new_bin["label"] > 0).astype(int)
    added_bin_counts = new_bin["label_binary"].value_counts().sort_index().to_dict()
    print(f"  new rows to add: {len(new_bin)}  (distribution: {added_bin_counts})")
    total_bin_rows = len(old_bin) + len(new_bin)
    print(f"  final row count: {total_bin_rows}")
    bin_label_new = pd.concat([old_bin["label"],
                               new_bin["label_binary"].rename("label")], ignore_index=True)
    print(f"  final class balance: {bin_label_new.value_counts().sort_index().to_dict()}")
    print("  new-row paths (spot check):")
    chk = path_exists_check(new_rows["file_path"])
    print(f"    unique new file_paths: {chk['total_unique']}")
    print(f"    sampled for existence check: {chk['sampled']}")
    print(f"    missing: {len(chk['missing_sample'])}")
    if chk["missing_sample"]:
        print(f"    first 3 missing: {chk['missing_sample'][:3]}")

    # 4. build 3-class-train preview
    old_3c = existing_train_3class_without_dep118()
    old_3c = remove_conflicting_rows(old_3c, existing_to_drop)
    hdr("4. 3-CLASS train preview (round2)")
    raw_3c_total = len(pd.read_csv(EXISTING_TRAIN_3CLASS))
    print(f"  existing train_3class.csv rows: {raw_3c_total}")
    print(f"  rows removed (dep-118 sound_id={DEP118_SOUND_ID_IN_SPLITS}): "
          f"{raw_3c_total - len(existing_train_3class_without_dep118())}")
    print(f"  rows removed (conflict option B): "
          f"{len(existing_train_3class_without_dep118()) - len(old_3c)}")
    # new rows where canonical label in {1, 2, 3} (whales only) — no noise into 3-class
    new_3c_src = new_rows[new_rows["label"].isin([1, 2, 3])].copy()
    new_3c_src["label_3c"] = new_3c_src["label"].map(CANONICAL_TO_3CLASS)
    print(f"  new rows to add (whales only): {len(new_3c_src)}")
    print(f"    by 3-class label: "
          f"{new_3c_src['label_3c'].value_counts().sort_index().to_dict()}")

    # humpback supplement — assigned globally unique sound_ids starting after new AL ids
    sid_offset = int(new_3c_src["sound_id"].max()) + 1 if len(new_3c_src) else SOUND_ID_START
    hump = sample_humpback_supplement(HUMPBACK_SUPPLEMENT_COUNT, sid_offset)
    print(f"  humpback supplement from {MULTISITE_TRAIN_3CLASS.name}:")
    print(f"    requested: {HUMPBACK_SUPPLEMENT_COUNT}  "
          f"actually sampled: {len(hump)}  "
          f"unique source audios: {hump['orig_audio_key'].nunique()}")
    if len(hump):
        print(f"    assigned global sound_ids: "
              f"{hump['sound_id'].min()}..{hump['sound_id'].max()}")
    if len(hump):
        print(f"    location breakdown: "
              f"{hump['location'].value_counts().to_dict()}")
    total_3c = len(old_3c) + len(new_3c_src) + len(hump)
    print(f"  final row count: {total_3c}")
    combined = pd.concat(
        [old_3c["label"], new_3c_src["label_3c"].rename("label"), hump["label"]],
        ignore_index=True,
    )
    print(f"  final class balance (0=Humpback, 1=Orca, 2=Beluga): "
          f"{combined.value_counts().sort_index().to_dict()}")

    # 5. val / test references
    hdr("5. Val / test references (no changes)")
    vb = pd.read_csv(VAL_BINARY)
    v3 = pd.read_csv(VAL_3CLASS)
    ft = pd.read_csv(FINAL_TEST_CSV)
    print(f"  binary val:  {VAL_BINARY.relative_to(REPO)}  rows={len(vb)}  "
          f"labels={vb['label'].value_counts().to_dict()}")
    print(f"  3class val:  {VAL_3CLASS.relative_to(REPO)}  rows={len(v3)}  "
          f"labels={v3['label'].value_counts().to_dict()}")
    print(f"  final test:  {FINAL_TEST_CSV.relative_to(REPO)}  rows={len(ft)}  "
          f"labels={ft['label'].value_counts().to_dict()}")

    # 6. overlap check — window-level duplicates in the FINAL round-2 manifests
    hdr("6. Leakage checks — final round-2 manifests only")
    audio_to_sid = _existing_audio_to_sid()
    # existing rows (cleaned) keyed by (sound_id, start, end)
    old_bin_keys = {(int(r["sound_id"]), int(r["start"]), int(r["end"]))
                    for _, r in old_bin.iterrows()}
    old_3c_keys = {(int(r["sound_id"]), int(r["start"]), int(r["end"]))
                   for _, r in old_3c.iterrows()}
    # new AL rows — translate audio → existing sound_id if it exists, else use new sound_id
    new_keys_bin = set()
    new_keys_3c = set()
    for _, r in new_rows.iterrows():
        eff_sid = audio_to_sid.get(r["audio"], int(r["sound_id"]))
        new_keys_bin.add((eff_sid, int(r["start"]), int(r["end"])))
        if int(r["label"]) in (1, 2, 3):
            new_keys_3c.add((eff_sid, int(r["start"]), int(r["end"])))
    bin_dup = old_bin_keys & new_keys_bin
    c3_dup  = old_3c_keys  & new_keys_3c
    print(f"  round-2 binary: old rows={len(old_bin_keys)}  new rows={len(new_keys_bin)}  "
          f"overlap={len(bin_dup)}")
    print(f"  round-2 3class: old rows={len(old_3c_keys)}  new rows={len(new_keys_3c)}  "
          f"overlap={len(c3_dup)}")
    if bin_dup:
        print(f"    bin overlap sample: {list(bin_dup)[:3]}")
    if c3_dup:
        print(f"    3c overlap sample: {list(c3_dup)[:3]}")
    # audio-level overlap (informational — expected since new AL covers some existing audios)
    def _strip_e(a): return a.replace(".e", "")
    new_audios_clean = set(new_rows["audio"].map(_strip_e))
    old_4c = pd.read_csv(EXISTING_TRAIN_4CLASS)
    old_4c["dep"] = old_4c["sound_path"].str.extract(r"/ID(\d+)/")[0]
    old_4c["audio_name"] = old_4c["sound_path"].str.extract(r"/(\d+)\.e\.wav$")[0]
    old_audios = set((old_4c["dep"] + "_" + old_4c["audio_name"]).dropna())
    overlap_audios = new_audios_clean & old_audios
    print(f"  audio-level overlap (informational): {len(overlap_audios)}  "
          f"{sorted(overlap_audios)[:5] if overlap_audios else ''}")

    hdr("7. Ready-to-write manifest targets (would be created next)")
    for p in [
        REPO / "data" / "tuxedni_splits" / "round2" / "train_binary.csv",
        REPO / "data" / "tuxedni_splits" / "round2" / "train_3class.csv",
    ]:
        print(f"  {p.relative_to(REPO)}  (exists: {p.exists()})")

    # 8. Phase-4 inference filter — preview the window keys that must be excluded
    hdr("8. Phase-4 inference-filter preview (round-2 inference on dep 120)")
    train_window_keys = _training_window_keys(old_bin, new_rows, old_3c,
                                              new_3c_src, hump, audio_to_sid)
    print(f"  training windows to exclude from round-2 inference: "
          f"{len(train_window_keys)} (matches by audio|start|end in SAMPLES)")
    print(f"  in-air audio range to also exclude: audio_num < {IN_AIR_MIN} "
          f"or > {IN_AIR_MAX}")
    print("  use filter_round2_inference(inference_df, train_window_keys) "
          "on the output of the new inference run")

    hdr("Summary of decisions")
    print(f"  seed (for humpback supplement): {SEED}")
    print(f"  humpback supplement: {HUMPBACK_SUPPLEMENT_COUNT} rows "
          f"from {MULTISITE_TRAIN_3CLASS.name}")
    print(f"  new sound_id range: {new_rows['sound_id'].min()}..{new_rows['sound_id'].max()}")
    print(f"  dep-118 removal: sound_id={DEP118_SOUND_ID_IN_SPLITS} dropped")
    print(f"  in-air filter: audio_num<{IN_AIR_MIN} or >{IN_AIR_MAX} (dep 120)")
    print(f"  conflict resolution: A={len(dropped_dups)} duplicates dropped, "
          f"B={len(existing_to_drop)} existing rows dropped")
    print("\nNo files written. Re-run after design changes; write step is separate.")


def _training_window_keys(
    old_bin: pd.DataFrame,
    new_rows: pd.DataFrame,
    old_3c: pd.DataFrame,
    new_3c_src: pd.DataFrame,
    hump: pd.DataFrame,
    audio_to_sid: dict,
) -> set:
    """Return the set of (audio_key, start, end) tuples that appear in training.

    Used to exclude these windows from round-2 inference on dep 120.
    audio_key is the 'DEP_NNNNNNNN.e' string used in the inference CSV's 'audio' column.
    We convert existing sound_id-based rows back to audio_key via the reverse map.
    """
    sid_to_audio = {v: k for k, v in audio_to_sid.items()}
    keys = set()
    # existing rows (have sound_id, start, end in samples)
    for df in (old_bin, old_3c):
        for _, r in df.iterrows():
            ak = sid_to_audio.get(int(r["sound_id"]))
            if ak is not None:
                keys.add((ak, int(r["start"]), int(r["end"])))
    # new AL rows (have audio, start, end in samples)
    for _, r in new_rows.iterrows():
        keys.add((r["audio"], int(r["start"]), int(r["end"])))
    # new 3c subset has same (audio, start, end) — subset of new_rows, already covered
    # humpback supplement: not on dep 120 (different locations), so no exclusion needed
    return keys


def filter_round2_inference(df: pd.DataFrame, train_window_keys: set) -> pd.DataFrame:
    """Apply round-2 inference filters (used AFTER the new inference CSV is produced).

    Keeps only dep-120 rows, excludes in-air audios, and excludes any window
    present in the round-2 training manifests.
    """
    df = df.copy()
    df["dep"] = df["audio"].str.split("_").str[0]
    df["audio_num"] = df["audio"].str.extract(r"120_(\d+)\.e")[0].astype("Int64")
    mask_dep = df["dep"] == "120"
    mask_in_air = (df["audio_num"] < IN_AIR_MIN) | (df["audio_num"] > IN_AIR_MAX)
    start_samp = (df["start(s)"] * SR).round().astype(int)
    end_samp = (df["end(s)"] * SR).round().astype(int)
    triples = list(zip(df["audio"], start_samp, end_samp))
    mask_train = pd.Series([t in train_window_keys for t in triples], index=df.index)
    return df.loc[mask_dep & ~mask_in_air & ~mask_train].drop(columns=["dep", "audio_num"])


if __name__ == "__main__":
    main()
