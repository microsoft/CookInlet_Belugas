"""
End-to-end cascade inference from raw audio, in one command.

This is a thin wrapper around ``inference.py``. Cascade mode in ``inference.py``
needs a folder of pre-computed ``.npy`` mel spectrograms; this script builds
those from raw ``.wav`` files first (using the same window + spectrogram
routines ``inference.py`` uses), then runs the two-stage (binary + 3-class)
cascade on them. Audio / spectrogram parameters are read from ``--config`` so
they stay identical to what the models were trained on.

Usage:
    python run_cascade.py \\
        --config data/data_config.yaml \\
        --audios_source /path/to/wavs \\
        --checkpoint_binary checkpoints/binary/best.ckpt \\
        --checkpoint_3class checkpoints/3class/best.ckpt \\
        --dataset mysite

The output CSV has the same columns as ``inference.py`` cascade mode; a Beluga
detection is ``pred_label == 3``. By default it writes to
``inference/<dataset>/cascade_results.csv``.

Spectrograms are cached under ``inference/<dataset>/spectrograms`` (override with
``--spectrograms_dir``). If that folder already holds ``.npy`` files they are
reused; pass ``--force`` to recompute.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path

from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import (
    build_inference_windows,
)
from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import (
    compute_mel_spectrograms_gpu,
)


def _cfg_get(cfg, *path, default=None):
    """Walk a dotted path into the config object, returning default if missing."""
    node = cfg
    for p in path:
        node = getattr(node, p, None)
        if node is None:
            return default
    return node


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end cascade inference from raw audio (wraps inference.py)."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="YAML config providing audio/spectrogram params (e.g. data/data_config.yaml)",
    )
    parser.add_argument(
        "--audios_source",
        required=True,
        help="Folder of .wav files (or a single .wav file)",
    )
    parser.add_argument("--checkpoint_binary", required=True, help="Stage-1 checkpoint")
    parser.add_argument("--checkpoint_3class", required=True, help="Stage-2 checkpoint")
    parser.add_argument(
        "--dataset",
        default=None,
        help="Name for the output folder (default: config 'name')",
    )
    parser.add_argument(
        "--output_csv",
        default=None,
        help="Cascade output CSV (default: inference/<dataset>/cascade_results.csv)",
    )
    parser.add_argument(
        "--spectrograms_dir",
        default=None,
        help="Where to write/read spectrograms (default: inference/<dataset>/spectrograms)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute spectrograms even if the folder already has .npy files",
    )

    # Passed through to inference.py cascade mode. Defaults match the shipped
    # checkpoints (and the manual): 224x180, temperature 3, normalize on.
    parser.add_argument(
        "--target_size", type=int, nargs=2, default=[224, 180], metavar=("H", "W")
    )
    parser.add_argument("--temperature", type=float, default=3.0)
    parser.add_argument(
        "--no_normalize",
        action="store_true",
        help="Disable spectrogram normalization (default: on)",
    )
    parser.add_argument("--conf_threshold", type=float, default=0.5)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")

    args = parser.parse_args()

    cfg = load_config(args.config)
    dataset = args.dataset or _cfg_get(cfg, "name", default="whales")

    sample_rate = _cfg_get(cfg, "audio", "sample_rate")
    window_size_sec = _cfg_get(cfg, "audio", "window_size_sec")
    overlap_sec = _cfg_get(cfg, "audio", "overlap_sec")
    n_fft = _cfg_get(cfg, "spectrogram", "n_fft")
    hop_length = _cfg_get(cfg, "spectrogram", "hop_length")
    n_mels = _cfg_get(cfg, "spectrogram", "n_mels")
    top_db = _cfg_get(cfg, "spectrogram", "top_db")
    noise_db_std = _cfg_get(cfg, "spectrogram", "noise_db_std", default=3.0)

    missing = [
        k
        for k, v in {
            "audio.sample_rate": sample_rate,
            "audio.window_size_sec": window_size_sec,
            "spectrogram.n_fft": n_fft,
            "spectrogram.hop_length": hop_length,
            "spectrogram.n_mels": n_mels,
        }.items()
        if v is None
    ]
    if missing:
        parser.error(f"Missing required config value(s): {missing} in {args.config}")

    spectrograms_dir = args.spectrograms_dir or os.path.join(
        "inference", dataset, "spectrograms"
    )
    os.makedirs(spectrograms_dir, exist_ok=True)

    # ---------------------------------------------------------------- #
    #  Step 1 — build mel spectrograms from raw audio (unless cached)
    # ---------------------------------------------------------------- #
    existing = list(Path(spectrograms_dir).glob("*.npy"))
    if existing and not args.force:
        print(
            f"[1/2] Found {len(existing)} existing spectrograms in "
            f"{spectrograms_dir}; skipping compute (use --force to recompute)."
        )
    else:
        print(f"[1/2] Building windows from {args.audios_source} ...")
        windows = build_inference_windows(
            audios_source=args.audios_source,
            window_size_sec=window_size_sec,
            overlap_sec=overlap_sec,
            sample_rate=sample_rate,
        )
        print(
            f"      {len(windows)} windows. Computing mel spectrograms -> "
            f"{spectrograms_dir}"
        )
        compute_mel_spectrograms_gpu(
            windows=windows,
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            top_db=top_db,
            spectrograms_path=spectrograms_dir,
            save_npy=True,
            fill_highfreq=True,
            noise_db_mean=None,
            noise_db_std=noise_db_std,
            storage_dtype="float32",
        )

    # ---------------------------------------------------------------- #
    #  Step 2 — run the cascade on those spectrograms via inference.py
    # ---------------------------------------------------------------- #
    output_csv = args.output_csv or os.path.join(
        "inference", dataset, "cascade_results.csv"
    )
    inference_py = os.path.join(os.path.dirname(os.path.abspath(__file__)), "inference.py")

    cmd = [
        sys.executable,
        inference_py,
        "--config", args.config,
        "--spectrograms_dir", spectrograms_dir,
        "--checkpoint_binary", args.checkpoint_binary,
        "--checkpoint_3class", args.checkpoint_3class,
        "--output_csv", output_csv,
        "--target_size", str(args.target_size[0]), str(args.target_size[1]),
        "--temperature", str(args.temperature),
        "--conf_threshold", str(args.conf_threshold),
        "--batch_size", str(args.batch_size),
        "--device", args.device,
        "--dataset", dataset,
    ]
    if not args.no_normalize:
        cmd.append("--normalize")

    print(f"[2/2] Running cascade -> {output_csv}")
    print("      " + " ".join(cmd))
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
