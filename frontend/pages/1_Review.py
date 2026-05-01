"""Spectrogram review & relabel UI (Component #2)."""

import io
import os
import sys
import time

import numpy as np
import streamlit as st

# Allow imports from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from csv_io import load_predictions, upsert_review, reviewed_path_for
from spec_render import render_spectrogram, CATEGORY_MAP
from audio_io import (load_audio_slice, parse_ear_log, get_recording_datetime,
                      compute_expanded_spectrogram, compute_2s_spectrogram,
                      apply_audio_processing)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_CSV = "/home/v-manoloc/shared/v-manoloc/tuxedni_results_1stAL_round_rev.csv"
INFERENCE_DIR = "/home/v-manoloc/shared/v-druizlopez/CookInlet_Belugas/inference"
MANUAL_VERIF_OPTIONS = ["", "Beluga", "Noise", "off_effort", "Humpback", "Orca"]

PLOT_GAIN_STEPS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="Review", page_icon="🔬", layout="wide")
st.title("🔬 Spectrogram Review")

# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Data source")

    available_csvs = [DEFAULT_CSV]
    if os.path.isdir(INFERENCE_DIR):
        for f in sorted(os.listdir(INFERENCE_DIR)):
            if f.endswith(".csv") and "threshold" not in f:
                p = os.path.join(INFERENCE_DIR, f)
                if p not in available_csvs:
                    available_csvs.append(p)

    selected_csv = st.selectbox(
        "Prediction CSV",
        available_csvs,
        index=0,
        format_func=os.path.basename,
    )

    st.divider()
    st.header("Filters")
    only_unverified = st.toggle("Only unverified rows", value=True)

    pred_label_filter = st.multiselect(
        "pred_label",
        options=list(CATEGORY_MAP.values()),
        default=list(CATEGORY_MAP.values()),
    )

    prob_range = st.slider("prob_whale (class 1) range", 0.0, 1.0, (0.0, 1.0), 0.01)

    st.divider()
    st.header("Display")
    freq_scale = st.radio("Frequency scale", ["Mel", "Linear Hz"], horizontal=True)
    spec_scale = "linear" if freq_scale == "Linear Hz" else "mel"

    view_mode = st.radio("Window", ["2 s", "10 s"], horizontal=True)
    expanded_view = view_mode == "10 s"

    plot_gain_idx = st.select_slider(
        "Plot gain",
        options=list(range(len(PLOT_GAIN_STEPS))),
        value=1,
        format_func=lambda i: f"{PLOT_GAIN_STEPS[i]:.1f}×",
    )
    plot_gain = PLOT_GAIN_STEPS[plot_gain_idx]

    auto_contrast = st.toggle("Auto-contrast", value=False)
    noise_reduction = st.toggle("Noise reduction", value=False)
    highpass = st.toggle("High-pass filter (≥300 Hz)", value=False)

    st.divider()
    st.header("Audio")
    playback_gain = st.slider("Playback gain", 0.1, 9.0, 2.5, 0.5)


# ── Load data ────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="Loading predictions…")
def _load(path):
    return load_predictions(path)


@st.cache_data(show_spinner=False)
def _load_ear_log():
    return parse_ear_log()


reviewed_path = reviewed_path_for(selected_csv)

# Use reviewed file if it already exists, otherwise start from source
load_path = reviewed_path if os.path.isfile(reviewed_path) else selected_csv
df_full = _load(load_path)

# ── Apply filters ─────────────────────────────────────────────────────────────

df = df_full.copy()

label_name_to_int = {v: k for k, v in CATEGORY_MAP.items()}
allowed_ints = [label_name_to_int[n] for n in pred_label_filter if n in label_name_to_int]
if allowed_ints:
    df = df[df["pred_label"].isin(allowed_ints)]

df = df[df["prob_class_1"].between(prob_range[0], prob_range[1])]

if only_unverified:
    df = df[df["manual_verif"].astype(str).str.strip() == ""]

total_unverified = (df_full["manual_verif"].astype(str).str.strip() == "").sum()
st.sidebar.metric("Total rows", len(df_full))
st.sidebar.metric("Unverified", int(total_unverified))
st.sidebar.metric("Matching filters", len(df))

# ── Session state — row index ─────────────────────────────────────────────────

if "row_idx" not in st.session_state or st.session_state.get("last_csv") != selected_csv:
    st.session_state["row_idx"] = 0
    st.session_state["last_csv"] = selected_csv

if len(df) == 0:
    st.info("No rows match the current filters. 🎉 All done, or adjust filters in the sidebar.")
    st.stop()

row_idx = min(st.session_state["row_idx"], len(df) - 1)
row = df.iloc[row_idx]

# ── Navigation ────────────────────────────────────────────────────────────────

st.markdown(
    f"<div style='font-size:1.1em'>Row <b>{row_idx + 1}</b> of <b>{len(df)}</b></div>",
    unsafe_allow_html=True,
)

# ── Jump to row / npy ────────────────────────────────────────────────────────

with st.expander("🔎 Jump to row or .npy file"):
    jump_col1, jump_col2 = st.columns(2)

    with jump_col1:
        jump_row = st.number_input(
            "Go to row number",
            min_value=1, max_value=len(df), value=row_idx + 1, step=1,
        )
        if st.button("Go to row"):
            st.session_state["row_idx"] = int(jump_row) - 1
            st.rerun()

    with jump_col2:
        npy_query = st.text_input("Search .npy filename (partial match)")
        if npy_query:
            matches = df[df["file_path"].str.contains(npy_query, case=False, na=False)]
            if not matches.empty:
                match_idx = df.index.get_loc(matches.index[0])
                st.caption(f"Found at row {match_idx + 1}: `{matches.iloc[0]['file_path'].split('/')[-1]}`")
                if st.button("Jump to match"):
                    st.session_state["row_idx"] = match_idx
                    st.rerun()
            else:
                st.caption("No match found.")

# ── Controls: Prev | Next | label radio | Save ────────────────────────────────

current_label = str(row["manual_verif"]).strip()
try:
    default_idx = MANUAL_VERIF_OPTIONS.index(current_label)
except ValueError:
    default_idx = 0

col_prev, col_next, col_radio, col_save, col_status = st.columns([1, 1, 6, 1, 2])
with col_prev:
    if st.button("← Prev", disabled=row_idx == 0):
        st.session_state["row_idx"] = max(0, row_idx - 1)
        st.rerun()
with col_next:
    if st.button("Next →", disabled=row_idx >= len(df) - 1):
        st.session_state["row_idx"] = min(len(df) - 1, row_idx + 1)
        st.rerun()
with col_radio:
    chosen = st.radio(
        "Assign label (`manual_verif`)",
        options=MANUAL_VERIF_OPTIONS,
        index=default_idx,
        horizontal=True,
        format_func=lambda x: x if x else "— unverified —",
        label_visibility="collapsed",
    )
with col_save:
    st.write("")  # vertical alignment nudge
    save_clicked = st.button("💾 Save", type="primary")
with col_status:
    status_placeholder = st.empty()

if save_clicked:
    save_df = load_predictions(reviewed_path if os.path.isfile(reviewed_path) else selected_csv)
    upsert_review(reviewed_path, str(row["file_path"]), chosen, save_df)
    _load.clear()
    ts = time.strftime("%H:%M:%S")
    status_placeholder.success(f"Saved at {ts}")
    if chosen and row_idx < len(df) - 1:
        st.session_state["row_idx"] = row_idx + 1
        st.rerun()

st.divider()

# ── Main layout ───────────────────────────────────────────────────────────────

col_spec, col_info = st.columns([3, 1])

with col_spec:
    @st.cache_resource(max_entries=200)
    def _figure(npy_path, pred_label, scale, auto_contrast, noise_reduction,
                plot_gain, expanded_view, highpass, audio_basename, start_s, end_s):
        if expanded_view:
            exp = compute_expanded_spectrogram(audio_basename, start_s, end_s, highpass=highpass)
            if exp is not None:
                return render_spectrogram(
                    npy_path, pred_label, scale=scale,
                    auto_contrast=auto_contrast, noise_reduction=noise_reduction,
                    plot_gain=plot_gain, highpass=highpass,
                    expanded_spec=exp["spec"],
                    t_markers=(exp["t_start"], exp["t_end"]),
                    t_total=exp["t_total"],
                )
        if highpass:
            filtered_spec = compute_2s_spectrogram(audio_basename, start_s, end_s, highpass=True)
            if filtered_spec is not None:
                return render_spectrogram(
                    npy_path, pred_label, scale=scale,
                    auto_contrast=auto_contrast, noise_reduction=noise_reduction,
                    plot_gain=plot_gain, highpass=highpass,
                    expanded_spec=filtered_spec,
                )
        return render_spectrogram(
            npy_path, pred_label, scale=scale,
            auto_contrast=auto_contrast, noise_reduction=noise_reduction,
            plot_gain=plot_gain, highpass=False,
        )

    try:
        fig = _figure(
            str(row["file_path"]), int(row["pred_label"]), spec_scale,
            auto_contrast, noise_reduction, plot_gain,
            expanded_view, highpass, str(row["audio"]),
            float(row["start(s)"]), float(row["end(s)"]),
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight")
        st.image(buf.getvalue(), use_container_width=True)
    except Exception as e:
        st.error(f"Could not render spectrogram: {e}")

with col_info:
    st.subheader("Probabilities")
    for label, col_key in [
        ("Beluga (cls 3)",   "prob_class_3"),
        ("Humpback (cls 1)", "prob_class_1"),
        ("Orca (cls 2)",     "prob_class_2"),
        ("Noise (cls 0)",    "prob_class_0"),
    ]:
        val = float(row.get(col_key, 0))
        st.caption(f"**{label}**: {val:.3f}")
        st.progress(val)
    st.markdown(f"**pred_label**: `{CATEGORY_MAP.get(int(row['pred_label']), row['pred_label'])}`")
    st.markdown(f"**segment_type**: `{row.get('segment_type', '—')}`")
    st.markdown(f"**window**: {row.get('start(s)', '?')}s – {row.get('end(s)', '?')}s")
    st.markdown(f"**audio**: `{row.get('audio', '—')}`")
    ear_log = _load_ear_log()
    rec_dt = get_recording_datetime(str(row.get("audio", "")), ear_log)
    if rec_dt:
        st.caption(f"🕐 Recording start: {rec_dt}")
    else:
        st.caption("🕐 Recording start: not found in EAR.LOG")

# ── Audio playback ────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False, max_entries=30)
def _raw_audio(audio_basename, start_s, end_s, use_expanded):
    """Return raw (data, sr) — cached per-row and per-view-mode."""
    if use_expanded:
        exp = compute_expanded_spectrogram(audio_basename, start_s, end_s)
        if exp is not None:
            return exp["audio"], exp["sr"]
    return load_audio_slice(audio_basename, start_s, end_s)

raw_data, raw_sr = _raw_audio(
    str(row["audio"]), float(row["start(s)"]), float(row["end(s)"]), expanded_view)

if raw_data is not None:
    import soundfile as _sf_out
    processed, out_sr = apply_audio_processing(raw_data, raw_sr, playback_gain, highpass, noise_reduction)
    buf = io.BytesIO()
    _sf_out.write(buf, processed, out_sr, format="WAV", subtype="FLOAT")
    st.audio(buf.getvalue(), format="audio/wav")
else:
    st.caption("⚠️ Audio unavailable for this row.")
