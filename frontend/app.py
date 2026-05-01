import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).parent))
from csv_io import DEFAULT_CSV, load_predictions, review_path_for

st.set_page_config(page_title="Bioacoustics Review", layout="wide")

with st.sidebar:
    st.header("Data")
    csv_path = st.text_input("Predictions CSV", value=DEFAULT_CSV, key="csv_path_input")

    reviewed = review_path_for(csv_path)
    if "df" not in st.session_state or st.session_state.get("loaded_csv") != csv_path:
        source = str(reviewed) if reviewed.exists() else csv_path
        st.session_state["df"] = load_predictions(source).copy()
        st.session_state["loaded_csv"] = csv_path
        st.session_state["reviewed_path"] = str(reviewed)
        st.session_state["row_idx"] = 0

    df = st.session_state["df"]
    if reviewed.exists():
        st.info(f"Resuming from `{reviewed.name}`")
    unverified = int((df["manual_verif"] == "").sum())
    st.success(f"{len(df):,} rows · {unverified:,} unverified")

st.title("Bioacoustics Pipeline Review")
st.markdown(
    """
Use the sidebar pages:
- **Review** — Step through predictions, listen to audio, assign labels.
- **AL Targets** — Check which thresholds meet your performance targets.
"""
)

cols = st.columns(4)
cols[0].metric("Total rows", f"{len(df):,}")
cols[1].metric("Unverified", f"{(df['manual_verif'] == '').sum():,}")
cols[2].metric("Reviewed", f"{(df['manual_verif'] != '').sum():,}")
cols[3].metric("Columns", len(df.columns))
