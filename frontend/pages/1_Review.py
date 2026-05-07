"""Spectrogram review & relabel UI.

Keyboard-first workflow: arrow keys to navigate, single-letter shortcuts
to assign a label (which auto-saves and advances to the next valid row),
1/2/3/4/p to toggle visualization options. The set of label buttons (and
their shortcuts) is configured in `frontend/config.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit_shortcuts

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from audio_io import (
    apply_audio_processing,
    compute_expanded_spectrogram,
    compute_segment_spectrogram,
    encode_wav,
    load_audio_slice,
)
from csv_io import backup_reviews, save_reviews
from ear_log import recording_date
from spec_render import render_spectrogram


# ── Visualization-toggle keyboard shortcuts ──────────────────────────────────

_VIS_SHORTCUTS = {
    "linear_scale": "4",
    "auto_contrast": "1",
    "noise_reduction": "3",
    "view_expanded": "2",
    "highpass": "p",
}

DEFAULTS = {
    "row_idx": 0,
    "save_count": 0,
    "view_expanded": False,
    "auto_contrast": False,
    "noise_reduction": False,
    "linear_scale": False,
    "highpass": False,
    "spec_gain": 1.0,
    "playback_gain": 1.0,
    "use_viridis": False,
}


# ── Bootstrap ────────────────────────────────────────────────────────────────

st.title("🔬 Review predictions")

if "df" not in st.session_state:
    st.warning("Load a CSV from the home page sidebar first.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
reviewed_path = Path(st.session_state["reviewed_path"])

for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Filtering ────────────────────────────────────────────────────────────────


def _outcome_predicates(df: pd.DataFrame):
    """Return (pred_pos, truth_pos) boolean Series, or (None, None) if the
    confusion-matrix filter isn't configured for this profile / CSV."""
    oc_pred = getattr(config, "OUTCOME_POSITIVE_PRED_VALUES", None)
    oc_truth_col = getattr(config, "OUTCOME_POSITIVE_TRUTH_COLUMN", None)
    oc_truth_vals = getattr(config, "OUTCOME_POSITIVE_TRUTH_VALUES", None)
    if (
        not oc_pred
        or not oc_truth_col
        or not oc_truth_vals
        or oc_truth_col not in df.columns
    ):
        return None, None
    pred_pos = df[config.PRED_LABEL_COLUMN].astype(int).isin(list(oc_pred))
    truth_pos = df[oc_truth_col].astype(int).isin(list(oc_truth_vals))
    return pred_pos, truth_pos


def _compute_valid_idx(
    df: pd.DataFrame,
    only_unverified: bool,
    pred_filter: list[str],
    prob_range: tuple[float, float] | None,
    outcome_filter: list[str] | None,
) -> list[int]:
    mask = pd.Series(True, index=df.index)
    if only_unverified:
        stage1 = df[config.MANUAL_VERIF_COLUMN].fillna("").astype(str)
        unverified = stage1 == ""
        stage2_col = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
        stage2_trigger = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
        if stage2_col and stage2_col in df.columns and stage2_trigger:
            stage2 = df[stage2_col].fillna("").astype(str)
            partial = (stage1 == stage2_trigger) & (stage2 == "")
            unverified = unverified | partial
        mask &= unverified
    label_to_int = {v: k for k, v in config.PRED_LABELS.items()}
    pred_ints = [label_to_int[name] for name in pred_filter if name in label_to_int]
    mask &= df[config.PRED_LABEL_COLUMN].astype(int).isin(pred_ints)
    bg_col = getattr(config, "BACKGROUND_PROB_COLUMN", None)
    if prob_range is not None and bg_col and bg_col in df.columns:
        prob_whale = 1.0 - df[bg_col].astype(float)
        lo, hi = prob_range
        mask &= (prob_whale >= lo) & (prob_whale <= hi)
    if outcome_filter:
        pred_pos, truth_pos = _outcome_predicates(df)
        if pred_pos is not None and truth_pos is not None:
            outcome_mask = pd.Series(False, index=df.index)
            if "TP" in outcome_filter:
                outcome_mask |= pred_pos & truth_pos
            if "FP" in outcome_filter:
                outcome_mask |= pred_pos & ~truth_pos
            if "TN" in outcome_filter:
                outcome_mask |= ~pred_pos & ~truth_pos
            if "FN" in outcome_filter:
                outcome_mask |= ~pred_pos & truth_pos
            mask &= outcome_mask
    return df.index[mask].tolist()


def _row_outcome(row, df: pd.DataFrame) -> str | None:
    """Compute TP/FP/TN/FN for a single row, or None if not configured."""
    oc_pred = getattr(config, "OUTCOME_POSITIVE_PRED_VALUES", None)
    oc_truth_col = getattr(config, "OUTCOME_POSITIVE_TRUTH_COLUMN", None)
    oc_truth_vals = getattr(config, "OUTCOME_POSITIVE_TRUTH_VALUES", None)
    if (
        not oc_pred
        or not oc_truth_col
        or not oc_truth_vals
        or oc_truth_col not in df.columns
    ):
        return None
    pred_pos = int(row[config.PRED_LABEL_COLUMN]) in oc_pred
    truth_pos = int(row[oc_truth_col]) in oc_truth_vals
    return ("T" if pred_pos == truth_pos else "F") + ("P" if pred_pos else "N")


# ── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Filters")
    only_unverified = st.checkbox("Only unverified", value=True)
    pred_filter = st.multiselect(
        "Predicted label",
        options=list(config.PRED_LABELS.values()),
        default=list(config.PRED_LABELS.values()),
    )
    _bg_col = getattr(config, "BACKGROUND_PROB_COLUMN", None)
    if _bg_col and _bg_col in df.columns:
        prob_range = st.slider(
            "P(non-background) range",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0),
            step=0.05,
            help=(
                "Two-handle range slider. Move LEFT handle up to keep only "
                "high-confidence detections; move RIGHT handle down to drop them."
            ),
        )
    else:
        prob_range = None

    _oc_pred = getattr(config, "OUTCOME_POSITIVE_PRED_VALUES", None)
    _oc_truth_col = getattr(config, "OUTCOME_POSITIVE_TRUTH_COLUMN", None)
    _oc_truth_vals = getattr(config, "OUTCOME_POSITIVE_TRUTH_VALUES", None)
    _outcome_enabled = bool(
        _oc_pred and _oc_truth_col and _oc_truth_vals and _oc_truth_col in df.columns
    )
    if _outcome_enabled:
        outcome_filter = st.multiselect(
            "Outcome (vs ground truth)",
            options=["TP", "FP", "TN", "FN"],
            default=["TP", "FP", "TN", "FN"],
            help=(
                "Binary cascade-level outcome. Positive prediction = "
                f"{config.PRED_LABEL_COLUMN} ∈ {sorted(_oc_pred or set())}; "
                f"positive truth = {_oc_truth_col} ∈ "
                f"{sorted(_oc_truth_vals or set())}."
            ),
        )
    else:
        outcome_filter = None

valid_idx = _compute_valid_idx(
    df, only_unverified, pred_filter, prob_range, outcome_filter
)

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
        f"Find by `{config.KEY_COLUMN}` (partial)",
        key="search_query",
        placeholder="e.g. 120_00038476",
    )
    if st.button("Find", key="btn_find") and query.strip():
        q = query.strip()
        hits = df.index[
            df[config.KEY_COLUMN].astype(str).str.contains(q, case=False, regex=False)
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
    st.radio(
        "Frequency scale",
        ["Mel", "Linear Hz"],
        horizontal=True,
        index=1 if st.session_state["linear_scale"] else 0,
        key="_freq_scale_radio",
    )
    st.session_state["linear_scale"] = (
        st.session_state["_freq_scale_radio"] == "Linear Hz"
    )
    st.checkbox("Auto-contrast (1)", key="auto_contrast")
    st.checkbox("Noise reduction (3)", key="noise_reduction")
    st.checkbox("Viridis colormap", key="use_viridis")
    st.slider(
        "Spectrogram gain",
        min_value=0.5,
        max_value=5.0,
        step=0.5,
        key="spec_gain",
    )
    streamlit_shortcuts.add_shortcuts(
        auto_contrast=_VIS_SHORTCUTS["auto_contrast"],
        noise_reduction=_VIS_SHORTCUTS["noise_reduction"],
    )

    st.subheader("Audio")
    st.checkbox(f"High-pass {int(config.HIGHPASS_CUTOFF_HZ)} Hz (p)", key="highpass")
    st.slider(
        "Playback gain",
        min_value=0.1,
        max_value=9.0,
        step=0.1,
        key="playback_gain",
    )
    streamlit_shortcuts.add_shortcuts(highpass=_VIS_SHORTCUTS["highpass"])

    if st.button("💾 Save now", use_container_width=True, key="btn_save_now"):
        save_reviews(reviewed_path, df)
        st.toast("Saved", icon="💾")

    st.subheader("Keyboard")
    label_cheats = "  \n".join(
        f"**{shortcut}** {label}" for label, shortcut in config.MANUAL_VERIF_LABELS
    )
    _s2_labels_cheat = getattr(config, "MANUAL_VERIF_STAGE2_LABELS", []) or []
    _s2_trigger_cheat = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
    if _s2_labels_cheat:
        s2_cheats = "  \n".join(f"**{sc}** {lbl}" for lbl, sc in _s2_labels_cheat)
        s2_block = f"_Subtype ({_s2_trigger_cheat}):_  \n{s2_cheats}  \n"
    else:
        s2_block = ""
    st.caption(
        "← / → Prev / Next  \n"
        f"{label_cheats}  \n"
        f"{s2_block}"
        "**2** expanded view · **1** auto-contrast · **3** noise red.  \n"
        "**p** highpass"
    )

    st.subheader("Session")
    sc = st.session_state["save_count"]
    backup_every = config.BACKUP_EVERY_N_SAVES
    next_backup_in = (backup_every - (sc % backup_every)) if sc > 0 else backup_every
    st.metric("Saves this session", sc, help=f"backup every {backup_every} saves")
    st.caption(f"next backup in {next_backup_in} more save(s)")
    if st.session_state.get("last_backup"):
        st.caption(f"last backup: `{st.session_state['last_backup']}`")
    st.caption(f"backups dir: `{reviewed_path.parent / 'backups'}`")


# ── Main panel ───────────────────────────────────────────────────────────────

audio_basename = str(row[config.AUDIO_COLUMN])
start_s = float(row[config.START_COLUMN])
end_s = float(row[config.END_COLUMN])

cmap = "viridis" if st.session_state["use_viridis"] else "magma"
freq_scale = "linear" if st.session_state["linear_scale"] else "mel"

# Spectrogram source: pre-computed .npy if no HPF and no expanded view,
# otherwise recompute from the source .wav with the current settings.
if st.session_state["view_expanded"]:
    expanded = compute_expanded_spectrogram(
        audio_basename, start_s, end_s, highpass=st.session_state["highpass"]
    )
    if expanded is not None:
        spec_arg = expanded["spec"]
        t_markers = (expanded["t_start"], expanded["t_end"])
        t_total = expanded["t_total"]
    else:
        spec_arg, t_markers, t_total = None, None, None
        st.caption("⚠️ wav not found for 10-s view — falling back to 2-s")
elif st.session_state["highpass"]:
    spec_arg = compute_segment_spectrogram(
        audio_basename, start_s, end_s, highpass=True
    )
    t_markers, t_total = None, None
else:
    spec_arg, t_markers, t_total = None, None, None

col_spec, col_meta = st.columns([3, 1])

with col_spec:
    fig = render_spectrogram(
        npy_path=str(row[config.SPEC_PATH_COLUMN]),
        pred_label=int(row[config.PRED_LABEL_COLUMN]),
        scale=freq_scale,
        auto_contrast=st.session_state["auto_contrast"],
        noise_reduction=st.session_state["noise_reduction"],
        spec_gain=st.session_state["spec_gain"],
        highpass=st.session_state["highpass"],
        expanded_spec=spec_arg,
        t_markers=t_markers,
        t_total=t_total,
        cmap=cmap,
    )
    st.pyplot(fig)

    # Audio: dual-player if expanded, single otherwise.
    def _audio_bytes_for(data, sr):
        processed, out_sr = apply_audio_processing(
            data,
            sr,
            playback_gain=st.session_state["playback_gain"],
            highpass=st.session_state["highpass"],
            noise_reduction=st.session_state["noise_reduction"],
        )
        return encode_wav(processed, out_sr)

    if st.session_state["view_expanded"]:
        col_a2, col_a10 = st.columns(2)
        with col_a2:
            st.caption(f"▶ {config.SEGMENT_VIEW_SEC:.0f}-s segment")
            d2, sr2 = load_audio_slice(audio_basename, start_s, end_s)
            if d2 is not None and sr2 is not None:
                st.audio(_audio_bytes_for(d2, sr2), format="audio/wav")
            else:
                st.caption("⚠️ Audio unavailable.")
        with col_a10:
            st.caption(f"▶ {config.EXPANDED_VIEW_SEC:.0f}-s window")
            exp = compute_expanded_spectrogram(audio_basename, start_s, end_s)
            if exp is not None:
                st.audio(_audio_bytes_for(exp["audio"], exp["sr"]), format="audio/wav")
            else:
                st.caption("⚠️ Audio unavailable.")
    else:
        d, sr = load_audio_slice(audio_basename, start_s, end_s)
        if d is not None and sr is not None:
            st.audio(_audio_bytes_for(d, sr), format="audio/wav")
        else:
            st.caption(f"⚠️ Audio unavailable for `{audio_basename}`")

    st.checkbox("10-second context view (key: 2)", key="view_expanded")
    streamlit_shortcuts.add_shortcuts(view_expanded=_VIS_SHORTCUTS["view_expanded"])

    date_str = recording_date(audio_basename)
    if date_str:
        st.caption(f"📅 Recording date: **{date_str}**")


with col_meta:
    st.subheader("Prediction")
    pred = int(row[config.PRED_LABEL_COLUMN])
    pred_label = config.PRED_LABELS.get(pred, "?")
    st.markdown(f"#### `{pred_label}`")

    for _gt_col, _gt_label, _gt_map in getattr(config, "GROUND_TRUTH_COLUMNS", []):
        if _gt_col in df.columns:
            try:
                _gt_int = int(row[_gt_col])
            except (TypeError, ValueError):
                continue
            _gt_text = _gt_map.get(_gt_int, str(_gt_int))
            st.markdown(f"**{_gt_label}**: `{_gt_text}`")

    _outcome_str = _row_outcome(row, df)
    if _outcome_str is not None:
        _outcome_color = {"TP": "🟢", "TN": "🟢", "FP": "🔴", "FN": "🔴"}.get(
            _outcome_str, "⚪"
        )
        st.markdown(f"**Outcome**: {_outcome_color} `{_outcome_str}`")

    current = (str(row[config.MANUAL_VERIF_COLUMN]) or "").strip()
    st.markdown(f"**Manual verif**: `{current or 'unverified'}`")

    btn_cols = st.columns(2)
    for i, (label, shortcut) in enumerate(config.MANUAL_VERIF_LABELS):
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
                df.at[row_idx, config.MANUAL_VERIF_COLUMN] = label
                save_reviews(reviewed_path, df)
                st.session_state["save_count"] += 1
                msg = f"Saved row {row_idx}: {label}"
                if st.session_state["save_count"] % config.BACKUP_EVERY_N_SAVES == 0:
                    backup = backup_reviews(reviewed_path)
                    st.session_state["last_backup"] = backup.name
                    msg += f"  · backup → {backup.name}"
                st.toast(msg, icon="✅")
                _s2_col = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
                _s2_trigger = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
                _needs_stage2 = (
                    _s2_col is not None
                    and _s2_trigger is not None
                    and label == _s2_trigger
                    and not str(df.at[row_idx, _s2_col]).strip()
                )
                if getattr(config, "AUTO_ADVANCE_ON_LABEL", True) and not _needs_stage2:
                    new_valid = _compute_valid_idx(
                        df, only_unverified, pred_filter, prob_range, outcome_filter
                    )
                    if row_idx not in new_valid:
                        next_after = next((i for i in new_valid if i > row_idx), None)
                        if next_after is not None:
                            st.session_state["row_idx"] = next_after
                st.rerun()

    if st.button(
        "Clear (back to unverified)", use_container_width=True, key="lbl_clear"
    ):
        df.at[row_idx, config.MANUAL_VERIF_COLUMN] = ""
        save_reviews(reviewed_path, df)
        st.toast(f"Cleared row {row_idx}", icon="🧹")
        st.rerun()

    # ── Stage 2 verification (hierarchical taxonomies) ───────────────────────
    _stage2_col = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
    _stage2_trigger = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
    _stage2_labels = getattr(config, "MANUAL_VERIF_STAGE2_LABELS", []) or []
    if (
        _stage2_col
        and _stage2_col in df.columns
        and current == _stage2_trigger
        and _stage2_labels
    ):
        st.divider()
        st.subheader(f"{_stage2_trigger}: subtype")
        current_s2 = (str(row[_stage2_col]) or "").strip()
        st.markdown(f"**Subtype**: `{current_s2 or 'unverified'}`")
        s2_btn_cols = st.columns(2)
        for i, (label2, shortcut2) in enumerate(_stage2_labels):
            col2 = s2_btn_cols[i % 2]
            with col2:
                btn_type2 = "primary" if label2 == current_s2 else "secondary"
                if streamlit_shortcuts.shortcut_button(
                    label2,
                    shortcut2,
                    hint=False,
                    type=btn_type2,
                    key=f"lbl2_{label2}",
                    use_container_width=True,
                ):
                    df.at[row_idx, _stage2_col] = label2
                    save_reviews(reviewed_path, df)
                    st.session_state["save_count"] += 1
                    msg = f"Saved row {row_idx} subtype: {label2}"
                    if (
                        st.session_state["save_count"] % config.BACKUP_EVERY_N_SAVES
                        == 0
                    ):
                        backup = backup_reviews(reviewed_path)
                        st.session_state["last_backup"] = backup.name
                        msg += f"  · backup → {backup.name}"
                    st.toast(msg, icon="✅")
                    if getattr(config, "AUTO_ADVANCE_ON_LABEL", True):
                        new_valid = _compute_valid_idx(
                            df,
                            only_unverified,
                            pred_filter,
                            prob_range,
                            outcome_filter,
                        )
                        if row_idx not in new_valid:
                            next_after = next(
                                (i for i in new_valid if i > row_idx), None
                            )
                            if next_after is not None:
                                st.session_state["row_idx"] = next_after
                    st.rerun()
        if st.button("Clear subtype", use_container_width=True, key="lbl2_clear"):
            df.at[row_idx, _stage2_col] = ""
            save_reviews(reviewed_path, df)
            st.toast(f"Cleared subtype on row {row_idx}", icon="🧹")
            st.rerun()

    st.subheader("Probabilities")
    for col_name, display_label, _ in config.PROB_BARS:
        if col_name in df.columns:
            p = float(row[col_name])
            st.progress(min(max(p, 0.0), 1.0), text=f"{display_label}: {p:.3f}")

    st.caption(f"audio: `{audio_basename}` · {start_s}s–{end_s}s")

# ── Bottom navigation ────────────────────────────────────────────────────────

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
        f"<div style='text-align:center'>Position <b>{pos + 1}</b> of "
        f"{len(valid_idx):,} <span style='color:#888'>(row {row_idx + 1} in CSV)</span></div>",
        unsafe_allow_html=True,
    )
