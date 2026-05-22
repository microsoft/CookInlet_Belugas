"""Spectrogram review & relabel UI.

Keyboard-first workflow: arrow keys to navigate, single-letter shortcuts
to assign a label (which auto-saves and advances to the next valid row),
1/2/3/4/p to toggle visualization options. The set of label buttons (and
their shortcuts) is configured in `frontend/config.py`.
"""

from __future__ import annotations

import bisect
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import streamlit_shortcuts

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from audio_io import (
    compute_expanded_spectrogram,
    compute_segment_spectrogram,
    get_expanded_wav_bytes,
    get_segment_wav_bytes,
)
from branding import render_logo
from prefs import init_preferences, save_preferences
from csv_io import BACKUP_DIR, save_backup, save_reviews
from ear_log import recording_date
from spec_render import render_spectrogram_png


# ── Visualization-toggle keyboard shortcuts ──────────────────────────────────

# Visualization/audio toggle keyboard shortcuts were removed: number keys
# conflicted with typing into the "Jump to position" field, and `p`
# collided with normal text input.

DEFAULTS = {
    "row_idx": 0,
    "save_count": 0,
    "unsaved_count": 0,
    "view_expanded": False,
    "auto_contrast": False,
    "noise_reduction": False,
    "linear_scale": False,
    "highpass": False,
    "spec_gain": 1.0,
    "playback_gain": 1.0,
    "use_viridis": False,
    "labeled_rows": [],  # recently labeled row indices kept in valid_idx
}


# ── Bootstrap ────────────────────────────────────────────────────────────────

render_logo()

st.markdown(
    """
    <style>
    /* Tighten gap below the spectrogram (x-axis label removed, only tick
       labels remain inside the figure, so a larger negative margin is safe). */
    div[data-testid="stPyplot"] { margin-bottom: -3rem; }
    /* Pull the audio player up toward the spectrogram; leave room below for
       the nav buttons. */
    div[data-testid="stAudio"] {
        margin-top: -1.25rem;
        margin-bottom: 0.25rem;
    }
    /* Tighten gap below the 5-column Prev/Next/counter nav row (two spacer
       columns flank the content for horizontal centering) but keep a bit
       of breathing room above it. */
    div[data-testid="stHorizontalBlock"]:has(
        > div[data-testid="stColumn"]:nth-child(5)
    ):not(:has(> div[data-testid="stColumn"]:nth-child(6))) {
        margin-top: 0;
        margin-bottom: -0.75rem;
    }
    /* Stage 2 (orca subtype) row: 7 columns. Tighten gaps + button padding
       and force single-line text so labels like "Unassigned" fit. */
    div[data-testid="stHorizontalBlock"]:has(
        > div[data-testid="stColumn"]:nth-child(7)
    ):not(:has(> div[data-testid="stColumn"]:nth-child(8))) {
        gap: 0.25rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(
        > div[data-testid="stColumn"]:nth-child(7)
    ):not(:has(> div[data-testid="stColumn"]:nth-child(8))) button {
        padding-left: 0.35rem !important;
        padding-right: 0.35rem !important;
    }
    div[data-testid="stHorizontalBlock"]:has(
        > div[data-testid="stColumn"]:nth-child(7)
    ):not(:has(> div[data-testid="stColumn"]:nth-child(8))) button p {
        white-space: nowrap;
        font-size: 0.85rem;
        margin: 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

if "df" not in st.session_state:
    st.warning("Load a CSV from the home page sidebar first.")
    st.stop()

df: pd.DataFrame = st.session_state["df"]
csv_path: str = st.session_state["loaded_csv"]
_rp_str = st.session_state.get("reviewed_path") or ""
reviewed_path: Path | None = Path(_rp_str) if _rp_str else None

NOTES_COLUMN = "notes"
if NOTES_COLUMN not in df.columns:
    df[NOTES_COLUMN] = ""


def _flush_notes(idx: int | None) -> None:
    """Commit the typed value of the notes text_area for `idx` back to df.

    Streamlit syncs widget values to session_state before any callback or
    rerun, so the typed text is available even if the user clicks Prev/Next
    without first clicking 'Save notes'. We mirror it into df here so the
    next save (auto-backup or manual) persists it.
    """
    if idx is None:
        return
    note_key = f"notes_{idx}"
    if note_key not in st.session_state:
        return
    cached_df = st.session_state.get("df")
    if cached_df is None or NOTES_COLUMN not in cached_df.columns:
        return
    new_val = st.session_state[note_key]
    current = cached_df.at[idx, NOTES_COLUMN]
    if pd.isna(current):
        current = ""
    if str(new_val) != str(current):
        cached_df.at[idx, NOTES_COLUMN] = new_val
        st.session_state["unsaved_count"] = st.session_state.get("unsaved_count", 0) + 1


init_preferences(DEFAULTS)


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


def _bool_filter(df: pd.DataFrame, col: str, choice: str) -> "pd.Series[bool]":
    """Tri-state mask for a boolean column. ``choice`` is "Any" / "True" /
    "False". If the column is missing, return all-True (no filter)."""
    if choice == "Any" or col not in df.columns:
        return pd.Series(True, index=df.index)
    truthy = df[col].astype(str).str.lower().isin(["true", "1", "yes"])
    return truthy if choice == "True" else ~truthy


def _compute_valid_idx(
    df: pd.DataFrame,
    only_unverified: bool,
    pred_filter: list[str],
    prob_range: tuple[float, float] | None,
    outcome_filter: list[str] | None,
    is_extra_choice: str = "Any",
    overlap_is_kw_choice: str = "Any",
    overlaps_csv_event_choice: str = "Any",
    keep_rows: list[int] | None = None,
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
        # Always keep recently labeled rows visible for navigation
        if keep_rows:
            for idx in keep_rows:
                if 0 <= idx < len(df):
                    unverified.iloc[idx] = True
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
    mask &= _bool_filter(df, "is_extra", is_extra_choice)
    mask &= _bool_filter(df, "overlap_is_KW", overlap_is_kw_choice)
    mask &= _bool_filter(df, "overlaps_csv_event", overlaps_csv_event_choice)
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

    # Boolean annotation-overlap filters (SMRU_extra profile carries these
    # columns; other profiles silently no-op when columns are absent).
    _bool_filter_cols = [
        ("is_extra", "is_extra (cascade KW with no CSV overlap)"),
        ("overlap_is_KW", "overlap_is_KW (window inside a KW CSV event)"),
        ("overlaps_csv_event", "overlaps_csv_event (window inside any CSV event)"),
    ]
    _bool_filter_choices: dict[str, str] = {}
    for _col, _label in _bool_filter_cols:
        if _col in df.columns:
            _bool_filter_choices[_col] = st.radio(
                _label,
                options=["Any", "True", "False"],
                index=0,
                horizontal=True,
                key=f"filter_{_col}",
            )
        else:
            _bool_filter_choices[_col] = "Any"

valid_idx = _compute_valid_idx(
    df,
    only_unverified,
    pred_filter,
    prob_range,
    outcome_filter,
    is_extra_choice=_bool_filter_choices["is_extra"],
    overlap_is_kw_choice=_bool_filter_choices["overlap_is_KW"],
    overlaps_csv_event_choice=_bool_filter_choices["overlaps_csv_event"],
    keep_rows=st.session_state.get("labeled_rows", []),
)

with st.sidebar:
    st.caption(f"{len(valid_idx):,} of {len(df):,} rows match filters")

_auto_advance = getattr(config, "AUTO_ADVANCE_ON_LABEL", True)
_current_row = st.session_state["row_idx"]

if not _auto_advance and _current_row not in valid_idx:
    bisect.insort(valid_idx, _current_row)

if not valid_idx:
    st.warning("No rows match the current filters.")
    st.stop()

if _current_row not in valid_idx:
    # Just update; do NOT call st.rerun() — that would abort the script
    # mid-sidebar and cause Streamlit to garbage-collect the visualization
    # widget keys before they render.
    st.session_state["row_idx"] = valid_idx[0]

row_idx = st.session_state["row_idx"]
row = df.iloc[row_idx]
pos = valid_idx.index(row_idx)

# Stash valid_idx so the Search/Jump callbacks (which fire before the script
# runs) can resolve positions and key-column hits without recomputing.
st.session_state["_valid_idx_cache"] = valid_idx


def _on_find_click():
    """Fired before the script reruns when the user clicks Find."""
    q = (st.session_state.get("search_query") or "").strip()
    if not q:
        return
    cached_df = st.session_state.get("df")
    if cached_df is None:
        return
    hits = cached_df.index[
        cached_df[config.KEY_COLUMN]
        .astype(str)
        .str.contains(q, case=False, regex=False)
    ].tolist()
    if hits:
        new_row = int(hits[0])
        _flush_notes(st.session_state.get("row_idx"))
        st.session_state["row_idx"] = new_row
        cached_valid = st.session_state.get("_valid_idx_cache", [])
        if new_row in cached_valid:
            st.session_state["_jump_input"] = cached_valid.index(new_row) + 1
        st.toast(f"Found {len(hits)} match(es); jumped to row {new_row}", icon="🔍")
    else:
        st.toast("No match", icon="⚠️")


def _on_jump_change():
    """Fired before the script reruns when the user changes the jump input."""
    new_jump = st.session_state.get("_jump_input")
    if new_jump is None:
        return
    cached = st.session_state.get("_valid_idx_cache", [])
    new_pos = int(new_jump) - 1
    if 0 <= new_pos < len(cached):
        _flush_notes(st.session_state.get("row_idx"))
        st.session_state["row_idx"] = cached[new_pos]


with st.sidebar:
    st.subheader("Search")
    st.text_input(
        f"Find by `{config.KEY_COLUMN}` (partial)",
        key="search_query",
        placeholder="e.g. 120_00038476",
    )
    st.button("Find", key="btn_find", on_click=_on_find_click)

    st.subheader("Navigation")
    # Apply any deferred jump-input value queued by Prev/Next on the previous
    # rerun. We must do this BEFORE instantiating the widget — Streamlit
    # forbids writing to a widget's session_state key after its widget has
    # been created in the current run.
    if "_pending_jump" in st.session_state:
        st.session_state["_jump_input"] = st.session_state.pop("_pending_jump")
    elif "_jump_input" not in st.session_state:
        st.session_state["_jump_input"] = pos + 1
    st.number_input(
        "Jump to position in filter",
        min_value=1,
        max_value=len(valid_idx),
        step=1,
        key="_jump_input",
        on_change=_on_jump_change,
    )

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
    st.checkbox("10-second context view", key="view_expanded")
    st.checkbox("Auto-contrast", key="auto_contrast")
    st.checkbox("Noise reduction", key="noise_reduction")
    st.checkbox("Viridis colormap", key="use_viridis")
    st.slider(
        "Spectrogram gain",
        min_value=0.5,
        max_value=5.0,
        step=0.5,
        key="spec_gain",
    )

    st.subheader("Audio")
    st.checkbox(f"High-pass {int(config.HIGHPASS_CUTOFF_HZ)} Hz", key="highpass")
    st.slider(
        "Playback gain",
        min_value=0.1,
        max_value=9.0,
        step=0.1,
        key="playback_gain",
    )

    unsaved = st.session_state["unsaved_count"]
    save_label = f"💾 Save now ({unsaved} unsaved)" if unsaved else "💾 Save now"
    if st.button(save_label, use_container_width=True, key="btn_save_now"):
        _flush_notes(st.session_state.get("row_idx"))
        new_path = save_reviews(csv_path, df)
        st.session_state["reviewed_path"] = str(new_path)
        reviewed_path = new_path
        st.session_state["unsaved_count"] = 0
        st.toast(f"Saved → `{new_path}`", icon="💾")

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
    st.caption(f"← / → Prev / Next  \n{label_cheats}  \n{s2_block}")

    st.subheader("Session")
    backup_every = config.BACKUP_EVERY_N_SAVES
    unsaved = st.session_state["unsaved_count"]
    next_auto = max(0, backup_every - unsaved)
    st.metric(
        "Unsaved verifications",
        unsaved,
        help=f"auto-backup fires every {backup_every} verifications",
    )
    st.caption(f"next auto-backup in {next_auto} more verification(s)")
    if st.session_state.get("last_backup"):
        st.caption(f"last auto-backup: `{st.session_state['last_backup']}`")
    st.caption(f"backups dir: `{BACKUP_DIR}`")


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

date_str = recording_date(audio_basename)

png_bytes = render_spectrogram_png(
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
    date_str=date_str,
)
st.image(png_bytes, use_container_width=True)

_playback_gain = st.session_state["playback_gain"]
_hpf = st.session_state["highpass"]
_nr = st.session_state["noise_reduction"]

if st.session_state["view_expanded"]:
    col_a2, col_a10 = st.columns(2)
    with col_a2:
        st.caption(f"▶ {config.SEGMENT_VIEW_SEC:.0f}-s segment")
        seg_bytes = get_segment_wav_bytes(
            audio_basename, start_s, end_s, _playback_gain, _hpf, _nr
        )
        if seg_bytes is not None:
            st.audio(seg_bytes, format="audio/wav")
        else:
            st.caption("⚠️ Audio unavailable.")
    with col_a10:
        st.caption(f"▶ {config.EXPANDED_VIEW_SEC:.0f}-s window")
        exp_bytes = get_expanded_wav_bytes(
            audio_basename, start_s, end_s, _playback_gain, _hpf, _nr
        )
        if exp_bytes is not None:
            with st.container(key="audio_active"):
                st.audio(exp_bytes, format="audio/wav")
        else:
            st.caption("⚠️ Audio unavailable.")
else:
    seg_bytes = get_segment_wav_bytes(
        audio_basename, start_s, end_s, _playback_gain, _hpf, _nr
    )
    if seg_bytes is not None:
        with st.container(key="audio_active"):
            st.audio(seg_bytes, format="audio/wav")
    else:
        st.caption(f"⚠️ Audio unavailable for `{audio_basename}`")

# Space-bar plays/pauses whichever audio is wrapped in `.st-key-audio_active`
# (the 10-s expanded player when expanded view is on, otherwise the 3-s
# segment). One-time install via a window-level flag so the listener isn't
# stacked on every fragment rerun.
components.html(
    """
    <script>
    (function() {
        const doc = window.parent.document;
        const w = window.parent.window;
        if (w.__spaceBarAudioInit) return;
        doc.addEventListener('keydown', function(e) {
            if (e.code !== 'Space' && e.key !== ' ') return;
            const t = e.target || {};
            const tag = (t.tagName || '').toLowerCase();
            if (tag === 'input' || tag === 'textarea' || t.isContentEditable) return;
            const audio = doc.querySelector('.st-key-audio_active audio');
            if (!audio) return;
            e.preventDefault();
            if (audio.paused) { audio.play(); } else { audio.pause(); }
        });
        w.__spaceBarAudioInit = true;

        // Suppress label shortcuts (o, e, etc.) while typing in inputs,
        // text_areas, or contentEditable fields. Capture phase + stop
        // propagation prevents `streamlit_shortcuts`' document-level listener
        // from ever seeing the keypress, so the character lands in the field.
        if (!w.__shortcutTypingGuard) {
            doc.addEventListener('keydown', function(e) {
                const t = e.target || {};
                const tag = (t.tagName || '').toLowerCase();
                if (tag === 'input' || tag === 'textarea' || t.isContentEditable) {
                    e.stopPropagation();
                }
            }, true);
            w.__shortcutTypingGuard = true;
        }
    })();
    </script>
    """,
    height=0,
    width=0,
)

_nav_pad_l, nav_prev, nav_next, nav_count, _nav_pad_r = st.columns(
    [1.5, 1, 1, 2.5, 1.5]
)
with nav_prev:
    if streamlit_shortcuts.shortcut_button(
        "← Prev",
        "arrowleft",
        hint=False,
        disabled=(pos == 0),
        use_container_width=True,
        key="btn_prev",
    ):
        _flush_notes(row_idx)
        st.session_state["row_idx"] = valid_idx[pos - 1]
        # Stash for the next rerun; we can't write _jump_input here because
        # the widget has already been instantiated up in the sidebar.
        st.session_state["_pending_jump"] = pos  # new pos = pos-1, display = pos
        st.rerun()
with nav_next:
    if streamlit_shortcuts.shortcut_button(
        "Next →",
        "arrowright",
        hint=False,
        disabled=(pos == len(valid_idx) - 1),
        use_container_width=True,
        key="btn_next",
    ):
        _flush_notes(row_idx)
        st.session_state["row_idx"] = valid_idx[pos + 1]
        st.session_state["_pending_jump"] = pos + 2
        st.rerun()
with nav_count:
    st.markdown(
        f"<div style='text-align:left; padding-top:0.4rem; padding-left:0.5rem'>"
        f"Position <b>{pos + 1}</b> of "
        f"{len(valid_idx):,} <span style='color:#888'>(row {row_idx + 2} in CSV)</span></div>",
        unsafe_allow_html=True,
    )


@st.fragment
def _verification_panel():
    """Bottom panel (prediction info + verification buttons).

    Wrapped in `@st.fragment` so clicking a label rerenders only this section
    — the spectrogram, audio bars, sidebar, and `_compute_valid_idx` (60k rows)
    are NOT re-executed. Only triggers a full-app rerun when `row_idx` changes
    (auto-advance), since the spectrogram lives outside this fragment.
    """
    col_pred, col_btn = st.columns(2)

    with col_pred:
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

        current = str(df.at[row_idx, config.MANUAL_VERIF_COLUMN]).strip()
        st.markdown(f"**Manual verif**: `{current or 'unverified'}`")

        if config.PROB_BARS:
            st.subheader("Probabilities")
            for col_name, display_label, _ in config.PROB_BARS:
                if col_name in df.columns:
                    raw = row[col_name]
                    if pd.isna(raw):
                        continue
                    p = float(raw)
                    st.progress(min(max(p, 0.0), 1.0), text=f"{display_label}: {p:.3f}")

        st.caption(f"sound file: `{audio_basename}` · {start_s}s–{end_s}s")

    with col_btn:
        st.subheader("Manual verification")
        btn_cols = st.columns(2)
        # When the stage-2 panel is visible, any stage-1 shortcut that collides
        # with a stage-2 label's shortcut yields to stage-2 (the user is in
        # subtype-selection mode), so we drop the keyboard binding on the
        # conflicting stage-1 button but keep it clickable.
        _s2_visible = (
            getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
            and getattr(config, "MANUAL_VERIF_STAGE2_COLUMN") in df.columns
            and current == getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
            and getattr(config, "MANUAL_VERIF_STAGE2_LABELS", [])
        )
        _s2_shortcut_set = (
            {sc for _, sc in getattr(config, "MANUAL_VERIF_STAGE2_LABELS", []) or []}
            if _s2_visible
            else set()
        )
        for i, (label, shortcut) in enumerate(config.MANUAL_VERIF_LABELS):
            col = btn_cols[i % 2]
            with col:
                btn_type = "primary" if label == current else "secondary"
                # streamlit_shortcuts persists bindings in a global JS map via
                # Object.assign — entries are never deleted, only overwritten by
                # button key. So a stale `lbl_Unsure: 'u'` from a prior render
                # (before stage-2 was visible) would shadow stage-2's `u` binding.
                # Re-register this same button key with a non-matching sentinel
                # to release the keystroke for stage-2.
                _effective_shortcut = (
                    "unbound" if shortcut in _s2_shortcut_set else shortcut
                )
                if streamlit_shortcuts.shortcut_button(
                    label,
                    _effective_shortcut,
                    hint=False,
                    type=btn_type,
                    key=f"lbl_{label}",
                    use_container_width=True,
                ):
                    df.at[row_idx, config.MANUAL_VERIF_COLUMN] = label
                    # If stage-1 moved off the stage-2 trigger (e.g. Orca →
                    # Bio), clear any stale stage-2 ecotype so it can't linger
                    # invisibly behind a non-Orca stage-1 label.
                    _s2c = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
                    _s2t = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
                    if _s2c and _s2c in df.columns and label != _s2t:
                        df.at[row_idx, _s2c] = ""
                    labeled = st.session_state.get("labeled_rows", [])
                    if row_idx not in labeled:
                        labeled.append(row_idx)
                    st.session_state["labeled_rows"] = labeled
                    st.session_state["unsaved_count"] += 1
                    msg = f"Row {row_idx + 2}: {label}"
                    if st.session_state["unsaved_count"] >= config.BACKUP_EVERY_N_SAVES:
                        backup = save_backup(csv_path, df)
                        st.session_state["last_backup"] = backup.name
                        st.session_state["unsaved_count"] = 0
                        msg += f"  · auto-backup → `{backup}`"
                    else:
                        msg += (
                            f"  · unsaved {st.session_state['unsaved_count']}/"
                            f"{config.BACKUP_EVERY_N_SAVES}"
                        )
                    st.toast(msg, icon="✅")
                    _s2_col = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
                    _s2_trigger = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
                    _needs_stage2 = (
                        _s2_col is not None
                        and _s2_trigger is not None
                        and label == _s2_trigger
                        and not str(df.at[row_idx, _s2_col]).strip()
                    )
                    if (
                        getattr(config, "AUTO_ADVANCE_ON_LABEL", True)
                        and not _needs_stage2
                    ):
                        new_valid = _compute_valid_idx(
                            df,
                            only_unverified,
                            pred_filter,
                            prob_range,
                            outcome_filter,
                            is_extra_choice=_bool_filter_choices["is_extra"],
                            overlap_is_kw_choice=_bool_filter_choices["overlap_is_KW"],
                            overlaps_csv_event_choice=_bool_filter_choices[
                                "overlaps_csv_event"
                            ],
                        )
                        if row_idx not in new_valid:
                            next_after = next(
                                (i for i in new_valid if i > row_idx), None
                            )
                            if next_after is not None:
                                _flush_notes(row_idx)
                                st.session_state["row_idx"] = next_after
                                st.rerun()  # full app rerun: row changed
                    # Row unchanged: rerun the fragment so the markdown and
                    # button colors pick up the new `current` value.
                    st.rerun(scope="fragment")

        _clear_col = btn_cols[len(config.MANUAL_VERIF_LABELS) % 2]
        with _clear_col:
            if st.button("Clear", use_container_width=True, key="lbl_clear"):
                df.at[row_idx, config.MANUAL_VERIF_COLUMN] = ""
                _s2c = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
                if _s2c and _s2c in df.columns:
                    df.at[row_idx, _s2c] = ""
                st.session_state["unsaved_count"] += 1
                st.toast(f"Cleared row {row_idx + 2} (unsaved)", icon="🧹")
                st.rerun(scope="fragment")

        _stage2_col = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)
        _stage2_trigger = getattr(config, "MANUAL_VERIF_STAGE2_TRIGGER", None)
        _stage2_labels = getattr(config, "MANUAL_VERIF_STAGE2_LABELS", []) or []
        if (
            _stage2_col
            and _stage2_col in df.columns
            and current == _stage2_trigger
            and _stage2_labels
        ):
            current_s2 = str(df.at[row_idx, _stage2_col]).strip()
            st.markdown(
                f"**{_stage2_trigger} subtype**: `{current_s2 or 'unverified'}`"
            )
            s2_btn_cols = st.columns(len(_stage2_labels), gap="small")
            for i, (label2, shortcut2) in enumerate(_stage2_labels):
                col2 = s2_btn_cols[i]
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
                        st.session_state["unsaved_count"] += 1
                        msg = f"Row {row_idx + 2} subtype: {label2}"
                        if (
                            st.session_state["unsaved_count"]
                            >= config.BACKUP_EVERY_N_SAVES
                        ):
                            backup = save_backup(csv_path, df)
                            st.session_state["last_backup"] = backup.name
                            st.session_state["unsaved_count"] = 0
                            msg += f"  · auto-backup → `{backup}`"
                        else:
                            msg += (
                                f"  · unsaved {st.session_state['unsaved_count']}/"
                                f"{config.BACKUP_EVERY_N_SAVES}"
                            )
                        st.toast(msg, icon="✅")
                        if getattr(config, "AUTO_ADVANCE_ON_LABEL", True):
                            new_valid = _compute_valid_idx(
                                df,
                                only_unverified,
                                pred_filter,
                                prob_range,
                                outcome_filter,
                                is_extra_choice=_bool_filter_choices["is_extra"],
                                overlap_is_kw_choice=_bool_filter_choices[
                                    "overlap_is_KW"
                                ],
                                overlaps_csv_event_choice=_bool_filter_choices[
                                    "overlaps_csv_event"
                                ],
                            )
                            if row_idx not in new_valid:
                                next_after = next(
                                    (i for i in new_valid if i > row_idx), None
                                )
                                if next_after is not None:
                                    _flush_notes(row_idx)
                                    st.session_state["row_idx"] = next_after
                                    st.rerun()  # full app rerun: row changed
                        # Row unchanged: refresh the fragment to reflect new subtype.
                        st.rerun(scope="fragment")
            if st.button("Clear subtype", use_container_width=True, key="lbl2_clear"):
                df.at[row_idx, _stage2_col] = ""
                st.session_state["unsaved_count"] += 1
                st.toast(f"Cleared subtype on row {row_idx + 2} (unsaved)", icon="🧹")
                st.rerun(scope="fragment")

        # Notes: free-text per-row field, keyed by row_idx so each row keeps
        # its own widget state. The typed value is mirrored into df by
        # `_flush_notes` whenever the user navigates (Prev/Next/jump/
        # auto-advance), so saving to disk picks it up even without clicking
        # "Save notes" first.
        _existing_note = df.at[row_idx, NOTES_COLUMN]
        if pd.isna(_existing_note):
            _existing_note = ""
        st.text_area(
            "Notes",
            value=str(_existing_note),
            key=f"notes_{row_idx}",
            placeholder="Free-text notes for this sample…",
            height=80,
        )
        if st.button("💾 Save notes", key="btn_save_notes"):
            _flush_notes(row_idx)
            new_path = save_reviews(csv_path, df)
            st.session_state["reviewed_path"] = str(new_path)
            st.session_state["unsaved_count"] = 0
            st.toast(f"Notes saved → `{Path(new_path).name}`", icon="📝")
            st.rerun(scope="fragment")


_verification_panel()

# Persist current visualization/audio settings so the next session opens with
# the same view. Cheap JSON write; runs once per full page rerun.
save_preferences()
