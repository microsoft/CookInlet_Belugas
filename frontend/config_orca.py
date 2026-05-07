"""Orca cascade-pipeline profile for the review UI.

Activated by setting `APP_PROFILE=orca` before launching streamlit. The base
`config` module re-exports this profile's values when the env var is set, so
no other code needs to know which profile is active.

Audio and spectrogram parameters are loaded from the orca repo's
`data/data_config.yaml` (path set via `ORCA_DATA_CONFIG` env var, defaulting
to the standard checkout location). This keeps the rendered y-axis frequency
labels and the HPF-recomputed spectrogram in sync with whatever values were
used to generate the cached `.npy` files — no manual mirroring.

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

import yaml


_ORCA_REPO_DEFAULT = Path("/home/v-druizlopez/bioacoustics/orcas_dclde2026")


def _env_path(name: str) -> Path | None:
    v = os.environ.get(name)
    return Path(v).expanduser() if v else None


def _load_yaml(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"Orca config not found at {path}. "
            f"Set the corresponding env var to its absolute path."
        )
    with path.open() as f:
        return yaml.safe_load(f)


_data_cfg = _load_yaml(
    _env_path("ORCA_DATA_CONFIG") or _ORCA_REPO_DEFAULT / "data" / "data_config.yaml"
)
_audio_cfg = _data_cfg["audio"]
_spec_cfg = _data_cfg["spectrogram"]

_3class_cfg = _load_yaml(
    _env_path("ORCA_3CLASS_CONFIG")
    or _ORCA_REPO_DEFAULT / "configs" / "config_3class.yaml"
)
_ecotype_cfg = _load_yaml(
    _env_path("ORCA_ECOTYPE_CONFIG")
    or _ORCA_REPO_DEFAULT / "configs" / "config_ecotype.yaml"
)
_3CLASS_NAMES: dict[int, str] = {
    int(k): v for k, v in _3class_cfg["class_names"].items()
}
_ECOTYPE_NAMES: dict[int, str] = {
    int(k): v for k, v in _ecotype_cfg["class_names"].items()
}


# ── Paths ────────────────────────────────────────────────────────────────────

AUDIO_ROOT: Path | None = _env_path("AUDIO_ROOT") or Path(
    "/home/v-druizlopez/shared/v-druizlopez/killer_whales_dclde2026"
)
INFERENCE_DIR: Path | None = _env_path("INFERENCE_DIR")
EAR_LOG_PATH: Path = _env_path("EAR_LOG_PATH") or (Path(__file__).parent / "EAR.LOG")
DEFAULT_CSV: str | None = os.environ.get("DEFAULT_CSV") or None


# ── Audio / spectrogram (sourced from data_config.yaml) ──────────────────────

SAMPLE_RATE: int = int(_audio_cfg["sample_rate"])
SEGMENT_VIEW_SEC: float = float(_audio_cfg["window_size_sec"])
N_FFT: int = int(_spec_cfg["n_fft"])
HOP_LENGTH: int = int(_spec_cfg["hop_length"])
N_MELS: int = int(_spec_cfg["n_mels"])
TOP_DB: float = float(_spec_cfg["top_db"])

# Review-tool knobs not present in data_config.yaml
HIGHPASS_CUTOFF_HZ: float = 300.0
EXPANDED_VIEW_SEC: float = 10.0
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


# ── Ground-truth display ─────────────────────────────────────────────────────
# Cascade test split carries per-window labels: a 3-class label and an
# ecotype label (sentinel -1 when no ecotype assignment). Class names come
# from the orca configs, so renaming a class in one place propagates here.

GROUND_TRUTH_COLUMNS: list[tuple[str, str, dict[int, str]]] = [
    ("label_3class", "GT 3-class", _3CLASS_NAMES),
    (
        "ecotype_label",
        "GT ecotype",
        {-1: "—", **_ECOTYPE_NAMES},
    ),
]


# ── Confusion-matrix filter (cascade-level orca detection) ───────────────────
# "Positive" = cascade emitted any KW class (Unassigned_KW or a specific
# ecotype). "Truth positive" = stage-1 ground truth is Orca. So:
#   TP: cascade said KW, label_3class==Orca
#   FP: cascade said KW, label_3class!=Orca
#   FN: cascade said NonBio/Bio, label_3class==Orca
#   TN: cascade said NonBio/Bio, label_3class!=Orca
# The cascade_output integer encoding lives in
# orcas_dclde2026/make_cascade_review_csv.py: {2..7} are the KW classes.

OUTCOME_POSITIVE_PRED_VALUES: set[int] | None = {2, 3, 4, 5, 6, 7}
OUTCOME_POSITIVE_TRUTH_COLUMN: str | None = "label_3class"
OUTCOME_POSITIVE_TRUTH_VALUES: set[int] | None = {2}


# ── Review-page UI knobs ─────────────────────────────────────────────────────

BACKUP_EVERY_N_SAVES: int = 5
LARGE_ROW_WARN: int = 10_000
