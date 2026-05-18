"""Streamlit entry point for the bioacoustics review app.

Picks the predictions CSV (auto-discovered from `INFERENCE_DIR` if set,
overridable via free-text path), loads it once into `session_state`, and
shows summary metrics. The Review and AL Targets pages then read from
`session_state` rather than re-loading.

To run:
conda activate orcas
cd /home/v-manoloc/CookInlet_Belugas
git checkout frontend
Then run:
export DEFAULT_CSV=/home/v-manoloc/orcas/cascade_predictions_for_review_manoloc.csv
export APP_PROFILE=orca
streamlit run /home/v-manoloc/CookInlet_Belugas/frontend/app.py

And open the Network URL, if you want in browser, go to port tab and click on globe symbol
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
import config
from branding import render_logo
from csv_io import latest_review_path, load_predictions

st.set_page_config(
    page_title="SPIRA",
    page_icon="🌀",
    layout="wide",
)


def _required_columns() -> list[str]:
    """Columns the review page reads directly from each row."""
    names = [
        "KEY_COLUMN",
        "SPEC_PATH_COLUMN",
        "AUDIO_COLUMN",
        "START_COLUMN",
        "END_COLUMN",
        "PRED_LABEL_COLUMN",
    ]
    seen: list[str] = []
    for name in names:
        col = getattr(config, name, None)
        if col and col not in seen:
            seen.append(col)
    return seen


def _csv_has_required_columns(path: Path) -> bool:
    required = _required_columns()
    if not required:
        return True
    try:
        with path.open() as f:
            header = [c.strip() for c in f.readline().rstrip("\n").split(",")]
    except OSError:
        return False
    return all(col in header for col in required)


def _discover_csvs() -> list[str]:
    """List candidate prediction CSVs from INFERENCE_DIR (header-validated)."""
    if config.INFERENCE_DIR is None or not config.INFERENCE_DIR.is_dir():
        return []
    return sorted(
        str(p)
        for p in config.INFERENCE_DIR.iterdir()
        if p.suffix == ".csv" and _csv_has_required_columns(p)
    )


with st.sidebar:
    st.header("Predictions CSV")

    discovered = _discover_csvs()
    options = list(discovered)
    default_idx: int | None = None
    if config.DEFAULT_CSV and os.path.isfile(config.DEFAULT_CSV):
        if config.DEFAULT_CSV not in options:
            options.insert(0, config.DEFAULT_CSV)
        default_idx = options.index(config.DEFAULT_CSV)

    if options:
        selected = st.selectbox(
            "Select",
            options=options,
            index=default_idx,
            format_func=lambda p: os.path.basename(p),
            placeholder="Choose a CSV…",
        )
    else:
        selected = None
        st.info(
            "Set `INFERENCE_DIR` and/or `DEFAULT_CSV` env vars, or paste a path below."
        )

    typed = st.text_input(
        "Or enter a path",
        value="",
        key="csv_path_input",
        placeholder="/path/to/predictions.csv",
    )
    csv_path = typed or selected or ""

    if not csv_path:
        st.stop()
    if not os.path.isfile(csv_path):
        st.error(f"Not a file: `{csv_path}`")
        st.stop()

    reviewed = latest_review_path(csv_path)

    # Guard: warn if loading the original when a reviewed version exists
    if reviewed and ("df" not in st.session_state or st.session_state.get("loaded_csv") != csv_path):
        st.warning(
            f"⚠️ A reviewed version already exists: **{reviewed.name}**\n\n"
            "Annotations will be loaded from the reviewed file automatically."
        )

    if "df" not in st.session_state or st.session_state.get("loaded_csv") != csv_path:
        source = str(reviewed) if reviewed else csv_path
        st.session_state["df"] = load_predictions(source).copy()
        st.session_state["loaded_csv"] = csv_path
        st.session_state["reviewed_path"] = str(reviewed) if reviewed else ""
        st.session_state["row_idx"] = 0

    df = st.session_state["df"]
    if reviewed:
        st.info(f"Resuming from `{reviewed.name}`")

    unverified = int((df[config.MANUAL_VERIF_COLUMN] == "").sum())
    st.success(f"{len(df):,} rows · {unverified:,} unverified")

render_logo()
st.markdown(
    """
    Open the **Review** page from the sidebar to step through spectrogram
    predictions, listen to audio, and assign labels.

    Reviewed labels are saved to a single file
    `frontend/reviews/<csv_stem>_<user>_<YYYYMMDDTHHMMSS>.csv` that is
    overwritten on every save. The source CSV is never modified. When you
    reload, the app automatically resumes from the reviewed file.
    """
)

cols = st.columns(4)
cols[0].metric("Total rows", f"{len(df):,}")
cols[1].metric("Unverified", f"{(df[config.MANUAL_VERIF_COLUMN] == '').sum():,}")
cols[2].metric("Reviewed", f"{(df[config.MANUAL_VERIF_COLUMN] != '').sum():,}")
cols[3].metric("Columns", len(df.columns))
