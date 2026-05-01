"""CSV load/save helpers for the review UI."""

import os
import tempfile

import pandas as pd

LARGE_ROW_WARN = 10_000


def _remap_path(path: str) -> str:
    """Remap file paths from Person A's home to the current shared mount."""
    return path.replace("/home/v-druizlopez/shared/", "/home/v-manoloc/shared/v-druizlopez/../")


def load_predictions(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "manual_verif" not in df.columns:
        df["manual_verif"] = ""
    else:
        df["manual_verif"] = df["manual_verif"].fillna("")
    if len(df) > LARGE_ROW_WARN:
        print(
            f"Warning: {len(df):,} rows loaded. Consider filtering before review."
        )
    return df


def upsert_review(reviewed_path: str, row_key: str, manual_verif_value: str, df: pd.DataFrame) -> None:
    """Write updated manual_verif for one row to the reviewed CSV (atomic write)."""
    mask = df["file_path"] == row_key
    df.loc[mask, "manual_verif"] = manual_verif_value

    dir_ = os.path.dirname(reviewed_path)
    os.makedirs(dir_, exist_ok=True)

    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".csv.tmp")
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, reviewed_path)
    except Exception:
        os.unlink(tmp)
        raise


def reviewed_path_for(input_csv: str) -> str:
    """Return the output path for this user's reviewed copy."""
    basename = os.path.splitext(os.path.basename(input_csv))[0]
    user = os.environ.get("USER", "unknown")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "frontend", "reviews", f"{basename}_{user}_reviewed.csv")


def autosave_path_for(input_csv: str) -> str:
    """Return the autosave (crash-backup) path for this user's session."""
    basename = os.path.splitext(os.path.basename(input_csv))[0]
    user = os.environ.get("USER", "unknown")
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, "frontend", "reviews", f"{basename}_{user}_autosave.csv")


def write_autosave(autosave_path: str, df: pd.DataFrame) -> None:
    """Overwrite the autosave backup (atomic write)."""
    dir_ = os.path.dirname(autosave_path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".csv.tmp")
    try:
        os.close(fd)
        df.to_csv(tmp, index=False)
        os.replace(tmp, autosave_path)
    except Exception:
        os.unlink(tmp)
        raise
