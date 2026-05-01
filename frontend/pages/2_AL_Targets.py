"""AL Targets tool (Component #3).

Load threshold-sweep CSVs, set min Precision / Recall / F1 sliders,
and see which thresholds (if any) meet your requirements.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="AL Targets", page_icon="🎯", layout="wide")
st.title("🎯 AL Targets")
st.markdown("Check whether any threshold meets your Precision / Recall / F1 requirements.")

# ── Discover sweep CSVs ───────────────────────────────────────────────────────

INFERENCE_DIR = "/home/v-manoloc/shared/v-druizlopez/CookInlet_Belugas/inference"

sweep_files = []
if os.path.isdir(INFERENCE_DIR):
    for f in sorted(os.listdir(INFERENCE_DIR)):
        if f.endswith(".csv") and "threshold_sweep" in f:
            sweep_files.append(os.path.join(INFERENCE_DIR, f))

if not sweep_files:
    st.error(f"No threshold-sweep CSVs found in `{INFERENCE_DIR}`.")
    st.stop()

# ── Sidebar — file selection + target sliders ─────────────────────────────────

with st.sidebar:
    st.header("Sweep files")
    selected_files = st.multiselect(
        "Select CSVs to evaluate",
        options=sweep_files,
        default=sweep_files,
        format_func=os.path.basename,
    )

    st.divider()
    st.header("Targets (Beluga class)")
    min_prec = st.slider("Min Beluga Precision", 0.0, 1.0, 0.85, 0.01)
    min_rec = st.slider("Min Beluga Recall", 0.0, 1.0, 0.70, 0.01)
    min_f1 = st.slider("Min Beluga F1", 0.0, 1.0, 0.0, 0.01)
    min_macro_f1 = st.slider("Min Macro F1", 0.0, 1.0, 0.0, 0.01)

if not selected_files:
    st.info("Select at least one sweep CSV in the sidebar.")
    st.stop()

# ── Evaluate each file ────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading sweep data…")
def _load_sweep(path):
    return pd.read_csv(path)


for fpath in selected_files:
    name = os.path.basename(fpath)
    df = _load_sweep(fpath)

    # Normalise column names (strip whitespace)
    df.columns = df.columns.str.strip()

    needed = {"threshold", "beluga_prec", "beluga_rec", "beluga_f1", "macro_f1"}
    missing_cols = needed - set(df.columns)
    if missing_cols:
        st.warning(f"**{name}**: missing columns {missing_cols} — skipped.")
        continue

    passing = df[
        (df["beluga_prec"] >= min_prec)
        & (df["beluga_rec"] >= min_rec)
        & (df["beluga_f1"] >= min_f1)
        & (df["macro_f1"] >= min_macro_f1)
    ].copy()

    st.subheader(f"📄 {name}")

    if not passing.empty:
        best = passing.loc[passing["beluga_f1"].idxmax()]
        st.success(
            f"✅ **{len(passing)} threshold(s) meet target** — "
            f"best at threshold **{best['threshold']:.2f}**: "
            f"Beluga P={best['beluga_prec']:.2f} "
            f"R={best['beluga_rec']:.2f} "
            f"F1={best['beluga_f1']:.2f} "
            f"| macro_F1={best['macro_f1']:.2f}"
        )
        st.dataframe(
            passing[["threshold", "beluga_prec", "beluga_rec", "beluga_f1", "macro_f1"]]
            .sort_values("beluga_f1", ascending=False)
            .reset_index(drop=True),
            use_container_width=True,
        )
    else:
        # Find closest row (highest beluga_f1 ignoring other constraints)
        closest = df.loc[df["beluga_f1"].idxmax()]
        st.error(
            f"❌ **No threshold meets target** — "
            f"closest is threshold **{closest['threshold']:.2f}**: "
            f"P={closest['beluga_prec']:.2f} "
            f"R={closest['beluga_rec']:.2f} "
            f"F1={closest['beluga_f1']:.2f} "
            f"| macro_F1={closest['macro_f1']:.2f}"
        )

    with st.expander("Show full sweep table"):
        st.dataframe(
            df[["threshold", "beluga_prec", "beluga_rec", "beluga_f1", "macro_f1"]]
            .style.highlight_between(
                subset=["beluga_prec"], left=min_prec, right=1.0, color="#1a472a"
            )
            .highlight_between(
                subset=["beluga_rec"], left=min_rec, right=1.0, color="#1a472a"
            ),
            use_container_width=True,
        )

    st.divider()
