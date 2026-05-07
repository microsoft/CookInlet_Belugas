"""Configuration for the bioacoustics review UI.

This is the only file you should need to edit when forking the app for a
different bioacoustic project. Paths are taken from environment variables
(so the same checkout works for any user); schema and class taxonomy are
constants you adapt to your model's outputs.

Environment variables (all optional; the app will degrade gracefully if a
path is unset or missing):

    AUDIO_ROOT      Directory holding source .wav files.
                    Required for audio playback and the 10-second view.
    INFERENCE_DIR   Directory holding prediction CSVs. Powers the
                    auto-discover dropdown on the home page.
    EAR_LOG_PATH    Optional text log mapping recording IDs to datetimes.
                    Defaults to frontend/EAR.LOG shipped with the repo.
    DEFAULT_CSV     Optional pre-selected CSV on launch.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


# ── Paths (machine/user-specific; do not hardcode) ───────────────────────────

AUDIO_ROOT: Path | None = _env_path("AUDIO_ROOT")
INFERENCE_DIR: Path | None = _env_path("INFERENCE_DIR")
EAR_LOG_PATH: Path = _env_path("EAR_LOG_PATH") or (Path(__file__).parent / "EAR.LOG")
DEFAULT_CSV: str | None = os.environ.get("DEFAULT_CSV") or None


# ── Audio / spectrogram (model-specific) ─────────────────────────────────────

SAMPLE_RATE: int = 24000
HIGHPASS_CUTOFF_HZ: float = 300.0
EXPANDED_VIEW_SEC: float = 10.0
SEGMENT_VIEW_SEC: float = 2.0
N_MELS: int = 128
TOP_DB: float = 80.0
PLAYBACK_SAMPLE_RATE: int = 44100  # browser-friendly resample target


# ── CSV schema (per-project) ─────────────────────────────────────────────────

KEY_COLUMN: str = "file_path"           # row identity (used as upsert key)
SPEC_PATH_COLUMN: str = "file_path"     # column holding the .npy spectrogram path
AUDIO_COLUMN: str = "audio"             # basename used to locate the .wav
START_COLUMN: str = "start(s)"
END_COLUMN: str = "end(s)"
PRED_LABEL_COLUMN: str = "pred_label"
MANUAL_VERIF_COLUMN: str = "manual_verif"
SEGMENT_TYPE_COLUMN: str = "segment_type"  # optional; shown in metadata


# ── Class taxonomy (per-project) ─────────────────────────────────────────────

PRED_LABELS: dict[int, str] = {
    0: "Background",
    1: "Humpback",
    2: "Orca",
    3: "Beluga",
}

# Title colour for the spectrogram (printed-on-white friendly)
PRED_TITLE_COLORS: dict[int, str] = {
    0: "#666666",
    1: "#b34700",
    2: "#5b0080",
    3: "#1a6b1a",
}

# Probability bar configuration. Order = display order under the spectrogram.
# Each entry: (csv_column, display_label, hex_color)
PROB_BARS: list[tuple[str, str, str]] = [
    ("prob_class_3", "Beluga",     "#1a6b1a"),
    ("prob_class_1", "Humpback",   "#b34700"),
    ("prob_class_2", "Orca",       "#5b0080"),
    ("prob_class_0", "Background", "#666666"),
]


# ── Manual-verification labels (per-project) ─────────────────────────────────
# Each entry: (label_text, single-character keyboard shortcut).
# The empty value "" is implicitly available as "unverified".

MANUAL_VERIF_LABELS: list[tuple[str, str]] = [
    ("Beluga",     "b"),
    ("Humpback",   "h"),
    ("Orca",       "o"),
    ("Noise",      "n"),
    ("off_effort", "z"),
    ("Unsure",     "u"),
]


# ── Review-page UI knobs ─────────────────────────────────────────────────────

BACKUP_EVERY_N_SAVES: int = 5
LARGE_ROW_WARN: int = 10_000
