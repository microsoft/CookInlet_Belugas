"""CSV load and save helpers for the review UI.

The source CSV is treated as read-only: reviews are written to a separate
file under `frontend/reviews/<csv_stem>_<user>_<YYYYMMDDTHHMMSS>.csv`. Each
save creates a NEW timestamped file (so the filename always tells you when
it was saved, and a directory listing is a save log). Periodic snapshots
land in `frontend/reviews/backups/` with the same naming convention.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import config

REVIEW_DIR = Path(__file__).parent / "reviews"
SAVE_MAX_ATTEMPTS = 5

_REVIEW_NAME = re.compile(r"^(?P<stem>.+)_(?P<user>[^_]+)_(?P<ts>\d{8}T\d{6})\.csv$")


def _now_ts() -> str:
    return datetime.now().strftime("%Y%m%dT%H%M%S")


def _user() -> str:
    return os.environ.get("USER", "anon")


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
    """Build a reviewed-CSV path for the given input + timestamp.

    Defaults `ts` to the current time. Same input + same second collapse
    to the same filename so rapid clicks within one second don't
    accumulate duplicate files.
    """
    if ts is None:
        ts = _now_ts()
    return REVIEW_DIR / f"{Path(input_csv).stem}_{_user()}_{ts}.csv"


def latest_review_path(input_csv: str) -> Path | None:
    """Most recent saved-reviews file for this input CSV+user, or None."""
    if not REVIEW_DIR.is_dir():
        return None
    user = _user()
    stem = Path(input_csv).stem
    candidates: list[tuple[str, Path]] = []
    for p in REVIEW_DIR.iterdir():
        m = _REVIEW_NAME.match(p.name)
        if m and m.group("stem") == stem and m.group("user") == user:
            candidates.append((m.group("ts"), p))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]


def save_reviews(input_csv: str, df: pd.DataFrame) -> Path:
    """Atomic-write the dataframe to a fresh timestamped path.

    Returns the path written to. Retries os.replace on transient OSErrors
    (EBUSY, locked files from sync agents / editors / antivirus).
    """
    target = review_path_for(input_csv)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".tmp_", suffix=".csv", dir=str(target.parent))
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        last_err: OSError | None = None
        for attempt in range(SAVE_MAX_ATTEMPTS):
            try:
                os.replace(tmp, target)
                return target
            except OSError as e:
                last_err = e
                if attempt + 1 < SAVE_MAX_ATTEMPTS:
                    time.sleep(0.1 * (2**attempt))
        if last_err is not None:
            raise last_err
        return target
    finally:
        if os.path.exists(tmp):
            try:
                os.unlink(tmp)
            except OSError:
                pass


def backup_reviews(input_csv: str, reviewed_path: Path) -> Path:
    """Copy the current reviewed CSV into `backups/` with a fresh timestamp."""
    if not reviewed_path.exists():
        raise FileNotFoundError(reviewed_path)
    backup_dir = REVIEW_DIR / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    backup_path = backup_dir / f"{Path(input_csv).stem}_{_user()}_{_now_ts()}.csv"
    shutil.copy2(reviewed_path, backup_path)
    return backup_path
