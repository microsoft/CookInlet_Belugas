"""CSV load and save helpers for the review UI.

The source CSV is treated as read-only. There are two save destinations:

  * `frontend/reviews/<csv_stem>_<user>_<YYYYMMDDTHHMMSS>.csv` —
    explicit "Save now" snapshots; multiple files accumulate (history).
  * `frontend/reviews/backups/<csv_stem>_<user>_<YYYYMMDDTHHMMSS>.csv` —
    auto-saved every N verifications; only the latest is kept (older
    auto-saves for the same input+user are deleted).

On launch, `latest_review_path` scans both directories and resumes from
whichever is newer.
"""

from __future__ import annotations

import os
import re
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config

REVIEW_DIR = Path(__file__).parent / "reviews"
BACKUP_DIR = REVIEW_DIR / "backups"
SAVE_MAX_ATTEMPTS = 5

_REVIEW_NAME = re.compile(r"^(?P<stem>.+)_(?P<user>[^_]+)_(?P<ts>\d{8}T\d{6})\.csv$")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _user() -> str:
    return os.environ.get("USER", "anon")


def _atomic_write_csv(df: pd.DataFrame, target: Path) -> None:
    """Write df to target atomically, retrying on transient OSErrors."""
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(target.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        last_err: OSError | None = None
        for attempt in range(SAVE_MAX_ATTEMPTS):
            try:
                os.replace(tmp, target)
                return
            except OSError as e:
                last_err = e
                if attempt + 1 < SAVE_MAX_ATTEMPTS:
                    time.sleep(0.1 * (2**attempt))
        if last_err is not None:
            raise last_err
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    """Load a predictions CSV. Adds an empty `manual_verif` column if missing."""
    df = pd.read_csv(path)
    verif_cols = [config.MANUAL_VERIF_COLUMN]
    if getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None):
        verif_cols.append(config.MANUAL_VERIF_STAGE2_COLUMN)
    for col in verif_cols:
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)
    if len(df) > config.LARGE_ROW_WARN:
        st.warning(
            f"{len(df):,} rows loaded. Consider filtering before review "
            f"(threshold: {config.LARGE_ROW_WARN:,})."
        )
    return df


def review_path_for(input_csv: str, ts: str | None = None) -> Path:
    """Path for an explicit save snapshot at the given timestamp."""
    if ts is None:
        ts = _now_ts()
    return REVIEW_DIR / f"{Path(input_csv).stem}_{_user()}_{ts}.csv"


def _matching_files(input_csv: str, dir_: Path) -> list[tuple[str, Path]]:
    """All files under dir_ matching <input_stem>_<user>_<ts>.csv."""
    if not dir_.is_dir():
        return []
    user = _user()
    stem = Path(input_csv).stem
    out: list[tuple[str, Path]] = []
    for p in dir_.iterdir():
        m = _REVIEW_NAME.match(p.name)
        if m and m.group("stem") == stem and m.group("user") == user:
            out.append((m.group("ts"), p))
    return out


def latest_review_path(input_csv: str) -> Path | None:
    """Most recent reviewed file (across reviews/ and reviews/backups/)."""
    candidates = _matching_files(input_csv, REVIEW_DIR) + _matching_files(
        input_csv, BACKUP_DIR
    )
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_reviews(input_csv: str, df: pd.DataFrame) -> Path:
    """Save an explicit snapshot to `reviews/`. Accumulates over time."""
    target = review_path_for(input_csv)
    _atomic_write_csv(df, target)
    return target


def save_backup(input_csv: str, df: pd.DataFrame) -> Path:
    """Auto-save to `reviews/backups/`, keeping at most one file per input+user.

    After writing the new backup, deletes any older backup files for the
    same input CSV / user so the directory holds a single most-recent
    auto-save.
    """
    target = BACKUP_DIR / f"{Path(input_csv).stem}_{_user()}_{_now_ts()}.csv"
    _atomic_write_csv(df, target)
    for _ts, p in _matching_files(input_csv, BACKUP_DIR):
        if p != target:
            try:
                p.unlink()
            except OSError:
                pass
    return target
