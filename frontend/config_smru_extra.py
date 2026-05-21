"""SMRU_extra LimeKiln-December profile for the review UI.

Activated via ``APP_PROFILE=smru_extra``. Differs from ``config_orca`` only
in (a) no per-window ground-truth columns (the SMRU set has no per-window
labels — only event-level verified positives), and (b) display columns
that surface the cascade's overlap with the LimeKiln_DecemberAnnotations
CSV so a reviewer can prioritise extras.

Run:
    export APP_PROFILE=smru_extra
    export ORCA_DATA_CONFIG=/home/v-druizlopez/bioacoustics/orcas_dclde2026/data/data_config.yaml
    export AUDIO_ROOT=/home/v-druizlopez/shared/v-druizlopez/killer_whales_dclde2026/SMRU_extra
    export INFERENCE_DIR=/home/v-druizlopez/bioacoustics/orcas_dclde2026/reports/smru_extra/manual_review
    export DEFAULT_CSV=$INFERENCE_DIR/review_for_frontend.csv
    streamlit run frontend/app.py --server.port 8501
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml


_ORCA_REPO_DEFAULT = Path("/home/v-druizlopez/bioacoustics/orcas_dclde2026")


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Config not found at {path}. Set the corresponding env var."
        )
    with path.open() as f:
        return yaml.safe_load(f)


_data_cfg = _load_yaml(
    _env_path("ORCA_DATA_CONFIG") or _ORCA_REPO_DEFAULT / "data" / "data_config.yaml"
)
_audio_cfg = _data_cfg["audio"]
_spec_cfg = _data_cfg["spectrogram"]


# ── Paths ────────────────────────────────────────────────────────────────────

AUDIO_ROOT: Path | None = _env_path("AUDIO_ROOT") or Path(
    "/home/v-druizlopez/shared/v-druizlopez/killer_whales_dclde2026/SMRU_extra"
)
INFERENCE_DIR: Path | None = _env_path("INFERENCE_DIR") or (
    _ORCA_REPO_DEFAULT / "reports" / "smru_extra" / "manual_review"
)
EAR_LOG_PATH: Path = _env_path("EAR_LOG_PATH") or (Path(__file__).parent / "EAR.LOG")
DEFAULT_CSV: str | None = os.environ.get("DEFAULT_CSV") or None


# ── Audio / spectrogram (sourced from data_config.yaml) ──────────────────────

SAMPLE_RATE: int = int(_audio_cfg["sample_rate"])
SEGMENT_VIEW_SEC: float = float(_audio_cfg["window_size_sec"])
N_FFT: int = int(_spec_cfg["n_fft"])
HOP_LENGTH: int = int(_spec_cfg["hop_length"])
N_MELS: int = int(_spec_cfg["n_mels"])
TOP_DB: float = float(_spec_cfg["top_db"])

HIGHPASS_CUTOFF_HZ: float = 300.0
EXPANDED_VIEW_SEC: float = 10.0
PLAYBACK_SAMPLE_RATE: int = 44100

_cascade_cfg = _data_cfg.get("cascade", {}) or {}
ECOTYPE_ABSTENTION_THRESHOLD: float = float(_cascade_cfg.get("threshold", 0.94))


# ── CSV schema ───────────────────────────────────────────────────────────────

KEY_COLUMN: str = "window_id"
SPEC_PATH_COLUMN: str = "file_path"
AUDIO_COLUMN: str = "audio"
START_COLUMN: str = "start(s)"
END_COLUMN: str = "end(s)"
PRED_LABEL_COLUMN: str = "pred_label"
MANUAL_VERIF_COLUMN: str = "manual_verif_3class"
SEGMENT_TYPE_COLUMN: str = "segment_type"


# ── Class taxonomy (mirrors config_orca) ─────────────────────────────────────

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
    0: "#666666",
    1: "#1a6b1a",
    2: "#5b0080",
    3: "#b34700",
    4: "#1f4e79",
    5: "#cc0066",
    6: "#006666",
    7: "#8b4513",
}

PROB_BARS: list[tuple[str, str, str]] = [
    ("prob_NonBio", "NonBio (stage 1)", "#666666"),
    ("prob_Bio", "Bio (stage 1)", "#1a6b1a"),
    ("prob_Orca", "Orca (stage 1)", "#5b0080"),
    ("prob_SRKW", "SRKW (stage 2 cal.)", "#b34700"),
    ("prob_TKW", "TKW (stage 2 cal.)", "#1f4e79"),
    ("prob_SAR", "SAR (stage 2 cal.)", "#cc0066"),
    ("prob_NRKW", "NRKW (stage 2 cal.)", "#006666"),
    ("prob_OKW", "OKW (stage 2 cal.)", "#8b4513"),
]
BACKGROUND_PROB_COLUMN: str | None = None


# ── Manual-verification labels ───────────────────────────────────────────────

MANUAL_VERIF_LABELS: list[tuple[str, str]] = [
    ("NonBio", "n"),
    ("Bio", "b"),
    ("Humpback", "h"),
    ("Orca", "o"),
    ("Unsure", "u"),
]

MANUAL_VERIF_STAGE2_COLUMN: str | None = "manual_verif_ecotype"
MANUAL_VERIF_STAGE2_TRIGGER: str | None = "Orca"
MANUAL_VERIF_STAGE2_LABELS: list[tuple[str, str]] = [
    ("SRKW", "s"),
    ("TKW", "t"),
    ("SAR", "a"),
    ("NRKW", "n"),
    ("OKW", "k"),
    ("Unassigned", "z"),
    ("Unsure", "u"),
]


# ── Ground-truth display ─────────────────────────────────────────────────────
# SMRU_extra has no per-window ground truth (the annotation CSV is
# event-level verified positives only). We surface the cascade↔annotation
# overlap status instead so the reviewer can see context:
#   overlaps_csv_event  — does this window fall inside any verified event?
#   overlap_event_tags  — which tags (KW_SRWK, HUMPBACK, FP_SHIPPROP, …)
#   overlap_is_KW       — is the overlapping event tagged KW_*?
GROUND_TRUTH_COLUMNS: list[tuple[str, str, dict[int, str]]] = []


# ── Confusion-matrix filter ──────────────────────────────────────────────────
# Disabled: with no per-window GT, the confusion matrix has no truth axis.
OUTCOME_POSITIVE_PRED_VALUES: set[int] | None = None
OUTCOME_POSITIVE_TRUTH_COLUMN: str | None = None
OUTCOME_POSITIVE_TRUTH_VALUES: set[int] | None = None


# ── Review-page UI knobs ─────────────────────────────────────────────────────

BACKUP_EVERY_N_SAVES: int = 5
LARGE_ROW_WARN: int = 10_000
AUTO_ADVANCE_ON_LABEL: bool = False
