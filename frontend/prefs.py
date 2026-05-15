"""Persistent visualization/audio preferences.

Saves a small JSON file in the user's home so that the review UI reopens with
the same Mel/Hz scale, auto-contrast, noise reduction, gain, etc. that the
user last chose. Keyed by `APP_PROFILE` so the orca profile and the default
beluga profile have independent preferences.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

# Session-state keys we persist. Keep in sync with the widgets in
# `pages/1_Review.py` (sidebar Visualization + Audio sections).
PREF_KEYS: list[str] = [
    "view_expanded",
    "auto_contrast",
    "noise_reduction",
    "linear_scale",
    "use_viridis",
    "spec_gain",
    "highpass",
    "playback_gain",
]


def _prefs_path() -> Path:
    profile = (os.environ.get("APP_PROFILE") or "default").strip().lower() or "default"
    return Path.home() / f".spiral_prefs_{profile}.json"


def _read_prefs() -> dict:
    path = _prefs_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def init_preferences(defaults: dict) -> None:
    """Seed session_state with saved preferences (or defaults when absent)."""
    saved = _read_prefs()
    for key, default in defaults.items():
        if key in st.session_state:
            continue
        st.session_state[key] = saved.get(key, default) if key in PREF_KEYS else default


def save_preferences() -> None:
    """Write the current visualization/audio settings to disk. Cheap (small
    file, atomic-enough for a single-user UI). Silently no-ops on I/O error."""
    snapshot = {
        key: st.session_state[key] for key in PREF_KEYS if key in st.session_state
    }
    try:
        _prefs_path().write_text(json.dumps(snapshot, indent=2))
    except OSError:
        pass
