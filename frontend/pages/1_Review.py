import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_shortcuts

sys.path.insert(0, str(Path(__file__).parent.parent))
from audio_io import load_audio_slice_wav
from csv_io import backup_reviews, save_reviews
from ear_log import recording_date
from spec_render import render_expanded_spectrogram, render_spectrogram

PRED_LABEL_MAP = {0: "No Whale", 1: "Humpback", 2: "Orca", 3: "Beluga"}
PRED_COLOR = {0: "gray", 1: "blue", 2: "red", 3: "green"}
PROB_LABELS = [
    ("prob_class_0", "No Whale"),
    ("prob_class_1", "Humpback"),
    ("prob_class_2", "Orca"),
    ("prob_class_3", "Beluga"),
]
LABELS_WITH_SHORTCUTS = [
    ("Beluga", "b"),
    ("Humpback", "h"),
    ("Orca", "o"),
    ("Noise", "n"),
    ("off_effort", "z"),
    ("Unsure", "u"),
]
BACKUP_EVERY = 5

DEFAULTS = {
    "row_idx": 0,
    "save_count": 0,
    "view_expanded": False,
    "auto_contrast": False,
    "noise_reduction": False,
    "linear_scale": False,
    "highpass": False,
    "plot_gain": 1.0,
    "playback_gain": 1.0,
}

st.title("Review predictions")

if "df" not in st.session_state:
    st.warning("Load a CSV from the home page sidebar first.")
    st.stop()

df = st.session_state["df"]
reviewed_path = Path(st.session_state["reviewed_path"])

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


def compute_valid_idx(df, only_unverified, pred_filter, prob_range):
    mask = pd.Series(True, index=df.index)
    if only_unverified:
        mask &= df["manual_verif"].fillna("").astype(str) == ""
    pred_ints = [k for k, v in PRED_LABEL_MAP.items() if v in pred_filter]
    mask &= df["pred_label"].astype(int).isin(pred_ints)
    prob_whale = 1.0 - df["prob_class_0"].astype(float)
    lo, hi = prob_range
    mask &= (prob_whale >= lo) & (prob_whale <= hi)
    return df.index[mask].tolist()


with st.sidebar:
    st.subheader("Filters")
    only_unverified = st.checkbox("Only unverified", value=True)
    pred_filter = st.multiselect(
        "Predicted label",
        options=list(PRED_LABEL_MAP.values()),
        default=list(PRED_LABEL_MAP.values()),
    )
    prob_range = st.slider(
        "P(whale) range",
        min_value=0.0,
        max_value=1.0,
        value=(0.0, 1.0),
        step=0.05,
        help=(
            "Range slider with two handles. Move LEFT handle up to keep only "
            "high-confidence whale rows; move RIGHT handle down to drop them."
        ),
    )

valid_idx = compute_valid_idx(df, only_unverified, pred_filter, prob_range)

with st.sidebar:
    st.caption(f"{len(valid_idx):,} of {len(df):,} rows match filters")

if not valid_idx:
    st.warning("No rows match the current filters.")
    st.stop()

if st.session_state["row_idx"] not in valid_idx:
    st.session_state["row_idx"] = valid_idx[0]
    st.rerun()

row_idx = st.session_state["row_idx"]
row = df.iloc[row_idx]
pos = valid_idx.index(row_idx)

with st.sidebar:
    st.subheader("Search")
    query = st.text_input(
        "Find by .npy filename (partial)",
        key="search_query",
        placeholder="e.g. 120_00038476",
    )
    if st.button("Find", key="btn_find") and query.strip():
        q = query.strip()
        hits = df.index[
            df["file_path"].astype(str).str.contains(q, case=False, regex=False)
        ].tolist()
        if hits:
            st.session_state["row_idx"] = int(hits[0])
            st.toast(f"Found {len(hits)} match(es); jumped to row {hits[0]}", icon="🔍")
            st.rerun()
        else:
            st.warning("No match")

    st.subheader("Navigation")
    jump = st.number_input(
        "Jump to position in filter",
        min_value=1,
        max_value=len(valid_idx),
        value=pos + 1,
        step=1,
    )
    if int(jump) - 1 != pos:
        st.session_state["row_idx"] = valid_idx[int(jump) - 1]
        st.rerun()

    st.subheader("Visualization")
    st.checkbox("Linear frequency (4)", key="linear_scale")
    st.checkbox("Auto-contrast (1)", key="auto_contrast")
    st.checkbox("Noise reduction (3)", key="noise_reduction")
    st.slider(
        "Plot gain",
        min_value=0.5,
        max_value=5.0,
        step=0.5,
        key="plot_gain",
    )
    streamlit_shortcuts.add_shortcuts(
        linear_scale="4",
        auto_contrast="1",
        noise_reduction="3",
    )

    st.subheader("Audio")
    st.checkbox("High-pass 500 Hz (p)", key="highpass")
    st.slider(
        "Playback gain",
        min_value=0.1,
        max_value=9.0,
        step=0.1,
        key="playback_gain",
    )
    streamlit_shortcuts.add_shortcuts(highpass="p")

    if st.button("💾 Save now", use_container_width=True, key="btn_save_now"):
        save_reviews(reviewed_path, df)
        st.toast("Saved", icon="💾")

    st.subheader("Keyboard")
    st.caption(
        "← / → Prev / Next  \n"
        "**b** Beluga · **h** Humpback · **o** Orca  \n"
        "**n** Noise · **z** off_effort · **u** Unsure  \n"
        "**2** 10s view · **1** auto-contrast · **3** noise red.  \n"
        "**4** linear/mel · **p** highpass"
    )

    st.subheader("Session")
    sc = st.session_state["save_count"]
    next_backup_in = (BACKUP_EVERY - (sc % BACKUP_EVERY)) if sc > 0 else BACKUP_EVERY
    st.metric("Saves this session", sc, help=f"backup every {BACKUP_EVERY} saves")
    st.caption(f"next backup in {next_backup_in} more save(s)")
    if st.session_state.get("last_backup"):
        st.caption(f"last backup: `{st.session_state['last_backup']}`")
    st.caption(f"backups dir: `{reviewed_path.parent / 'backups'}`")

date_str = recording_date(str(row["audio"]))

col_spec, col_meta = st.columns([3, 1])

spec_kwargs = dict(
    auto_contrast=st.session_state["auto_contrast"],
    noise_reduction=st.session_state["noise_reduction"],
    linear_scale=st.session_state["linear_scale"],
    plot_gain=st.session_state["plot_gain"],
)

with col_spec:
    if st.session_state["view_expanded"]:
        fig = render_expanded_spectrogram(
            str(row["audio"]),
            float(row["start(s)"]),
            float(row["end(s)"]),
            **spec_kwargs,
        )
        if fig is None:
            st.caption("⚠️ wav not found for 10s view — falling back to 2s")
            fig = render_spectrogram(row["file_path"], **spec_kwargs)
    else:
        fig = render_spectrogram(row["file_path"], **spec_kwargs)
    st.pyplot(fig)

    audio_bytes = load_audio_slice_wav(
        str(row["audio"]),
        float(row["start(s)"]),
        float(row["end(s)"]),
        expanded=st.session_state["view_expanded"],
        gain=st.session_state["playback_gain"],
        highpass=st.session_state["highpass"],
    )
    if audio_bytes is None:
        st.caption(f"⚠️ audio file not found for `{row['audio']}`")
    else:
        st.audio(audio_bytes, format="audio/wav")

    st.checkbox("10-second context view (key: 2)", key="view_expanded")
    streamlit_shortcuts.add_shortcuts(view_expanded="2")

    if date_str:
        st.caption(f"📅 Recording date: **{date_str}**")
    else:
        st.caption("📅 Recording date: *not in EAR.LOG*")

with col_meta:
    st.subheader("Prediction")
    pred = int(row["pred_label"])
    pred_label = PRED_LABEL_MAP.get(pred, "?")
    pred_color = PRED_COLOR.get(pred, "gray")
    st.markdown(f"#### :{pred_color}[**{pred_label}**]")

    current = (row["manual_verif"] or "").strip()
    st.markdown(f"**Manual verif**: `{current or 'unverified'}`")

    btn_cols = st.columns(2)
    for i, (label, shortcut) in enumerate(LABELS_WITH_SHORTCUTS):
        col = btn_cols[i % 2]
        with col:
            btn_type = "primary" if label == current else "secondary"
            if streamlit_shortcuts.shortcut_button(
                label,
                shortcut,
                hint=False,
                type=btn_type,
                key=f"lbl_{label}",
                use_container_width=True,
            ):
                df.at[row_idx, "manual_verif"] = label
                save_reviews(reviewed_path, df)
                st.session_state["save_count"] += 1
                msg = f"Saved row {row_idx}: {label}"
                if st.session_state["save_count"] % BACKUP_EVERY == 0:
                    backup = backup_reviews(reviewed_path)
                    st.session_state["last_backup"] = backup.name
                    msg += f"  · backup → {backup.name}"
                st.toast(msg, icon="✅")
                new_valid = compute_valid_idx(
                    df, only_unverified, pred_filter, prob_range
                )
                if row_idx not in new_valid:
                    next_after = next((i for i in new_valid if i > row_idx), None)
                    if next_after is not None:
                        st.session_state["row_idx"] = next_after
                st.rerun()

    if st.button(
        "Clear (back to unverified)", use_container_width=True, key="lbl_clear"
    ):
        df.at[row_idx, "manual_verif"] = ""
        save_reviews(reviewed_path, df)
        st.toast(f"Cleared row {row_idx}", icon="🧹")
        st.rerun()

    st.subheader("Probabilities")
    for col, label in PROB_LABELS:
        p = float(row[col])
        st.progress(min(max(p, 0.0), 1.0), text=f"{label}: {p:.3f}")

    st.caption(f"audio: `{row['audio']}` · {row['start(s)']}s–{row['end(s)']}s")

st.divider()
col_p, col_c, col_n = st.columns([1, 2, 1])
with col_p:
    if streamlit_shortcuts.shortcut_button(
        "← Prev",
        "arrowleft",
        hint=False,
        disabled=(pos == 0),
        use_container_width=True,
        key="btn_prev",
    ):
        st.session_state["row_idx"] = valid_idx[pos - 1]
        st.rerun()
with col_n:
    if streamlit_shortcuts.shortcut_button(
        "Next →",
        "arrowright",
        hint=False,
        disabled=(pos == len(valid_idx) - 1),
        use_container_width=True,
        key="btn_next",
    ):
        st.session_state["row_idx"] = valid_idx[pos + 1]
        st.rerun()
with col_c:
    st.markdown(
        f"<div style='text-align:center'>Position <b>{pos + 1}</b> of {len(valid_idx):,} "
        f"<span style='color:#888'>(row {row_idx + 1} in CSV)</span></div>",
        unsafe_allow_html=True,
    )
