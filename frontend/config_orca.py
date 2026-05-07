"""Orca cascade-pipeline profile for the review UI.

Activated by setting `APP_PROFILE=orca` before launching streamlit. The base
`config` module re-exports this profile's values when the env var is set, so
no other code needs to know which profile is active.

Two-stage manual verification:
  Stage 1 (3-class):   NonBio / Bio / Orca / Unsure
  Stage 2 (ecotype):   SRKW / TKW / SAR / NRKW / OKW / Unassigned / Unsure
                       (only enabled when stage 1 == "Orca")

Ground truth and predictions live in the joined CSV produced by
`orcas_dclde2026/make_cascade_review_csv.py`.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


# ── Paths ────────────────────────────────────────────────────────────────────

AUDIO_ROOT: Path | None = _env_path("AUDIO_ROOT") or Path(
    "/home/v-druizlopez/shared/v-druizlopez/killer_whales_dclde2026"
)
INFERENCE_DIR: Path | None = _env_path("INFERENCE_DIR")
EAR_LOG_PATH: Path = _env_path("EAR_LOG_PATH") or (Path(__file__).parent / "EAR.LOG")
DEFAULT_CSV: str | None = os.environ.get("DEFAULT_CSV") or None


# ── Audio / spectrogram ──────────────────────────────────────────────────────

SAMPLE_RATE: int = 16000
HIGHPASS_CUTOFF_HZ: float = 300.0
EXPANDED_VIEW_SEC: float = 10.0
SEGMENT_VIEW_SEC: float = 3.0
N_MELS: int = 128
TOP_DB: float = 80.0
PLAYBACK_SAMPLE_RATE: int = 44100


# ── CSV schema ───────────────────────────────────────────────────────────────

KEY_COLUMN: str = "window_id"
SPEC_PATH_COLUMN: str = "file_path"
AUDIO_COLUMN: str = "audio"
START_COLUMN: str = "start(s)"
END_COLUMN: str = "end(s)"
PRED_LABEL_COLUMN: str = "pred_label"
MANUAL_VERIF_COLUMN: str = "manual_verif_3class"
SEGMENT_TYPE_COLUMN: str = "segment_type"


# ── Class taxonomy: cascade output (8 buckets) ───────────────────────────────

PRED_LABELS: dict[int, str] = {
    0: "NonBio",
    1: "Bio",
    2: "Unassigned_KW",
    3: "SRKW",
    4: "TKW",
    5: "SAR",
    6: "NRKW",
    7: "OKW",
}

PRED_TITLE_COLORS: dict[int, str] = {
    0: "#666666",  # NonBio — grey
    1: "#1a6b1a",  # Bio — green
    2: "#5b0080",  # Unassigned_KW — purple
    3: "#b34700",  # SRKW — orange
    4: "#1f4e79",  # TKW — navy
    5: "#cc0066",  # SAR — magenta
    6: "#006666",  # NRKW — teal
    7: "#8b4513",  # OKW — brown
}

# The cascade CSV stores only the calibrated max-confidence; per-class
# probabilities are not saved by cascade_eval.py. Keep PROB_BARS empty so the
# Review page skips the probability slider and the per-class bar list.
PROB_BARS: list[tuple[str, str, str]] = []
BACKGROUND_PROB_COLUMN: str | None = None


# ── Stage 1: 3-class verification (top-level manual_verif column) ────────────

MANUAL_VERIF_LABELS: list[tuple[str, str]] = [
    ("NonBio", "n"),
    ("Bio", "b"),
    ("Orca", "o"),
    ("Unsure", "u"),
]


# ── Stage 2: ecotype verification (only when stage 1 == "Orca") ──────────────

MANUAL_VERIF_STAGE2_COLUMN: str | None = "manual_verif_ecotype"
MANUAL_VERIF_STAGE2_TRIGGER: str | None = "Orca"
MANUAL_VERIF_STAGE2_LABELS: list[tuple[str, str]] = [
    ("SRKW", "s"),
    ("TKW", "t"),
    ("SAR", "a"),
    ("NRKW", "r"),
    ("OKW", "k"),
    ("Unassigned", "z"),
    ("Unsure", "q"),
]


# ── Review-page UI knobs ─────────────────────────────────────────────────────

BACKUP_EVERY_N_SAVES: int = 5
LARGE_ROW_WARN: int = 10_000
