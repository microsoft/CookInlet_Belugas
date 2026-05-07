"""CSV load and save helpers for the review UI.

The source CSV is treated as read-only: reviews are written to a separate
file under `frontend/reviews/<csv_stem>_<user>_reviewed.csv` so concurrent
reviewers on different clones never stomp each other. A timestamped copy is
written to `frontend/reviews/backups/` every `BACKUP_EVERY_N_SAVES` saves.
"""

from __future__ import annotations

import os
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


def review_path_for(input_csv: str) -> Path:
    """Per-user reviewed-CSV destination."""
    user = os.environ.get("USER", "anon")
    return REVIEW_DIR / f"{Path(input_csv).stem}_{user}_reviewed.csv"


def save_reviews(reviewed_path: Path, df: pd.DataFrame) -> None:
    """Atomic-write the full reviewed dataframe.

    Retries the os.replace step on transient OSErrors (EBUSY, locked
    files from sync agents / editors / antivirus). On final failure
    cleans up the tmp file and re-raises so the caller can surface a
    user-visible toast.
    """
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_", suffix=".csv", dir=str(reviewed_path.parent)
    )
    os.close(fd)
    try:
        df.to_csv(tmp, index=False)
        last_err: OSError | None = None
        for attempt in range(SAVE_MAX_ATTEMPTS):
            try:
                os.replace(tmp, reviewed_path)
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


def backup_reviews(reviewed_path: Path) -> Path:
    """Copy the current reviewed CSV into `backups/` with a timestamp."""
    if not reviewed_path.exists():
        raise FileNotFoundError(reviewed_path)
    backup_dir = reviewed_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"{reviewed_path.stem}_{ts}.csv"
    shutil.copy2(reviewed_path, backup_path)
    return backup_path
