import os
import shutil
import tempfile
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

DEFAULT_CSV = "/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/inference/tuxedni_results_1stAL_round_rev.csv"
REVIEW_DIR = Path(
    "/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/frontend/reviews"
)


@st.cache_data(show_spinner=False)
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "manual_verif" not in df.columns:
        df["manual_verif"] = ""
    df["manual_verif"] = df["manual_verif"].fillna("").astype(str)
    return df


def review_path_for(input_csv: str) -> Path:
    user = os.environ.get("USER", "anon")
    return REVIEW_DIR / f"{Path(input_csv).stem}_{user}_reviewed.csv"


def save_reviews(reviewed_path: Path, df: pd.DataFrame) -> None:
    reviewed_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        prefix=".tmp_", suffix=".csv", dir=str(reviewed_path.parent)
    )
    os.close(fd)
    df.to_csv(tmp, index=False)
    os.rename(tmp, reviewed_path)


def backup_reviews(reviewed_path: Path) -> Path:
    if not reviewed_path.exists():
        raise FileNotFoundError(reviewed_path)
    backup_dir = reviewed_path.parent / "backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = backup_dir / f"{reviewed_path.stem}_{ts}.csv"
    shutil.copy2(reviewed_path, backup_path)
    return backup_path
