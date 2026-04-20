"""
Usage:
    # Binary inference
    python inference.py --checkpoint model.ckpt --audios_source /path/to/audio --dataset birds

    # Multiclass inference
    python inference.py --checkpoint model.ckpt --audios_source /path/to/audio --dataset whales \
        --num_classes 3 --class_names "Humpback,Orca,Beluga"

    # Using config file
    python inference.py --config data/data_config.yaml --checkpoint model.ckpt --audios_source /path/to/audio
"""

import os
import argparse
import re
import json
import math
from pathlib import Path
from collections import defaultdict
from typing import Optional, List, Dict, Union

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torchaudio


def spectrogram_filename(sound_path: str, start: int, end: int) -> str:
    """Return the .npy filename for a spectrogram window (standard convention)."""
    sound_filename = os.path.splitext(os.path.basename(sound_path))[0]
    return f"{sound_filename}_{int(start)}_{int(end)}.npy"


def resolve_spectrogram_path(
    spectrograms_dir: str,
    *,
    sound_path: Optional[str] = None,
    start: int = 0,
    end: int = 0,
    sound_id: Optional[int] = None,
    window_id: Optional[int] = None,
    label: Optional[int] = None,
) -> str:
    """Return the full path of an existing spectrogram, checking both naming conventions.

    Naming conventions
    ------------------
    Standard : ``{sound_filename}_{start}_{end}.npy``
    Legacy   : ``sid{sound_id}_idx{window_id}_start{start}_end{end}_lab{label}.npy``

    The function checks whether the standard file exists first.  If not it
    falls back to the legacy format (when enough metadata is provided).  When
    neither exists it returns the standard path so that new files are always
    created under the standard convention.
    """
    # Standard name
    if sound_path is not None:
        std_name = spectrogram_filename(sound_path, start, end)
    else:
        std_name = None

    if std_name is not None:
        std_full = os.path.join(spectrograms_dir, std_name)
        if os.path.exists(std_full):
            return std_full

    # Legacy name (requires sound_id, window_id and label)
    if sound_id is not None and window_id is not None and label is not None:
        legacy_name = (
            f"sid{sound_id}_idx{window_id}"
            f"_start{start}_end{end}_lab{label}.npy"
        )
        legacy_full = os.path.join(spectrograms_dir, legacy_name)
        if os.path.exists(legacy_full):
            return legacy_full

    # Default to standard (used for new files)
    if std_name is not None:
        return os.path.join(spectrograms_dir, std_name)
    return os.path.join(spectrograms_dir, f"unknown_{start}_{end}.npy")

import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import librosa
import soundfile as sf

# Import from PytorchWildlife core library
from PytorchWildlife.models.bioacoustics import ResNetClassifier, load_model_from_checkpoint
from PytorchWildlife.data.bioacoustics.bioacoustics_datasets import ResizeTo, PerSampleNormalize, BioacousticsInferenceDataset
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_inference_windows
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config
from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import compute_mel_spectrograms_gpu


def build_dataframe_from_spectrograms_dir(
    spectrograms_dir: str,
    sample_rate: int = 24000,
) -> pd.DataFrame:
    """Build a DataFrame from a directory of pre-computed .npy spectrograms.

    Filename convention: {audio_name}_{start_samples}_{end_samples}.npy
    """
    npy_files = sorted(Path(spectrograms_dir).glob("*.npy"))
    rows = []
    for f in npy_files:
        parts = f.stem.split("_")
        start_samples = int(parts[-2])
        end_samples = int(parts[-1])
        audio = "_".join(parts[:-2])
        rows.append({
            'file_path': str(f),
            'audio': audio,
            'start(s)': start_samples / sample_rate,
            'end(s)': end_samples / sample_rate,
        })
    return pd.DataFrame(rows)


def run_inference_batch(
    model: ResNetClassifier,
    dataloader: DataLoader,
    sample_rate: int,
    num_classes: int = 2,
    annotations_json: Optional[str] = None,
    device: str = "cuda",
    conf_threshold: float = 0.5,
    temperature: float = 1.0,
    meta_df: Optional[pd.DataFrame] = None,
) -> Dict[str, np.ndarray]:
    """
    Run inference on a batch of data. Supports both binary and multiclass.
    """
    is_binary = (num_classes == 2)
    model.eval()
    all_paths = []
    all_logits = []

    print(f"Running inference on {len(dataloader)} batches...")
    print(f"Mode: {'binary' if is_binary else f'multiclass ({num_classes} classes)'}")

    with torch.no_grad():
        for batch_idx, batch in tqdm(enumerate(dataloader), total=len(dataloader)):
            x, paths = batch
            x = x.to(device)

            logits = model(x)
            if is_binary:
                logits = logits.squeeze(1)
            all_logits.append(logits.cpu().numpy())
            all_paths.extend(paths)

    # Parse audio paths, starts, and ends
    if meta_df is not None:
        audios = meta_df['audio'].tolist()
        starts = meta_df['start(s)'].tolist()
        ends = meta_df['end(s)'].tolist()
    else:
        annotations = None
        if annotations_json is not None:
            with open(annotations_json, "r") as f:
                annotations = json.load(f)

        audios = []
        starts = []
        ends = []
        for p in all_paths:
            if "start" in p and "end" in p:
                try:
                    sound_id = int(re.search(r'sid(\d+)_', p).group(1))
                    if annotations:
                        audios.append(next(s["file_name_path"] for s in annotations["sounds"] if s["id"] == sound_id))
                    else:
                        audios.append(f"sound_{sound_id}")
                    starts.append(float(re.search(r'start(\d+)_end', p).group(1)) / sample_rate)
                    ends.append(float(re.search(r'end(\d+)\_lab', p).group(1)) / sample_rate)
                except (AttributeError, StopIteration):
                    basename = os.path.basename(p).replace(".npy", "")
                    parts = basename.split("_")
                    audios.append("_".join(parts[:-2]))
                    starts.append(int(parts[-2]) / sample_rate)
                    ends.append(int(parts[-1]) / sample_rate)
            else:
                basename = os.path.basename(p).replace(".npy", "")
                parts = basename.split("_")
                audios.append("_".join(parts[:-2]))
                starts.append(int(parts[-2]) / sample_rate)
                ends.append(int(parts[-1]) / sample_rate)

    all_logits = np.concatenate(all_logits)

    if is_binary:
        scaled_logits = all_logits / temperature
        probabilities = 1 / (1 + np.exp(-scaled_logits))
        predictions = (probabilities > conf_threshold).astype(int)
    else:
        logits_tensor = torch.tensor(all_logits) / temperature
        probabilities = F.softmax(logits_tensor, dim=1).numpy()
        predictions = probabilities.argmax(axis=1)

    return {
        'paths': all_paths,
        'audios': audios,
        'starts': starts,
        'ends': ends,
        'predictions': predictions,
        'probabilities': probabilities,
    }


def process_inference_results_per_second(csv_path: str) -> pd.DataFrame:
    """
    Process inference results CSV and obtain a prediction for each second,
    averaging the predictions that overlap according to the start(s) and end(s) columns.
    """
    df = pd.read_csv(csv_path)
    unique_audios = df['audio'].unique()

    all_results = []

    for audio in unique_audios:
        audio_df = df[df['audio'] == audio].copy()

        min_start = int(np.floor(audio_df['start(s)'].min()))
        max_end = int(np.ceil(audio_df['end(s)'].max()))

        for second in range(min_start, max_end):
            overlapping = audio_df[
                ((audio_df['start(s)'] <= second) & (audio_df['end(s)'] > second)) |
                ((audio_df['start(s)'] < second + 1) & (audio_df['end(s)'] >= second + 1))
            ]

            if len(overlapping) > 0:
                weights = []
                for _, row in overlapping.iterrows():
                    overlap_start = max(row['start(s)'], second)
                    overlap_end = min(row['end(s)'], second + 1)
                    overlap_duration = max(0, overlap_end - overlap_start)
                    weights.append(overlap_duration)

                weights = np.array(weights)

                if weights.sum() > 0:
                    weights = weights / weights.sum()

                    avg_prediction = np.average(overlapping['prediction'], weights=weights)
                    avg_probability = np.average(overlapping['probability'], weights=weights)
                    avg_confidence = np.average(overlapping['confidence'], weights=weights)

                    all_results.append({
                        'audio': audio,
                        'second': second,
                        'count_overlaps': len(overlapping),
                        'prediction': 1 if avg_prediction >= 0.5 else 0,
                        'avg_prediction': avg_prediction,
                        'avg_probability': avg_probability,
                        'avg_confidence': avg_confidence,
                    })

    results_df = pd.DataFrame(all_results)
    results_df = results_df.sort_values(['audio', 'second']).reset_index(drop=True)

    output_dir = os.path.dirname(csv_path)
    output_path = os.path.join(output_dir, 'per_second_results.csv')

    results_df.to_csv(output_path, index=False)
    print(f"Per-second results saved to: {output_path}")

    return results_df


def save_inference_results(
    results: Dict,
    output_path: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
) -> pd.DataFrame:
    """Save inference results to CSV in appropriate format."""
    is_binary = (num_classes == 2)

    if is_binary:
        results_df = pd.DataFrame({
            'audio': results['audios'],
            'start(s)': results['starts'],
            'end(s)': results['ends'],
            'prediction': results['predictions'],
            'probability': results['probabilities'],
            'confidence': np.abs(results['probabilities'] - 0.5) * 2,
        })
        results_df = results_df.sort_values('confidence', ascending=False)
    else:
        data = {
            'file_path': results['paths'],
            'audio': results['audios'],
            'start(s)': results['starts'],
            'end(s)': results['ends'],
            'prediction': results['predictions'],
        }

        if class_names is None:
            class_names = [f"class_{i}" for i in range(num_classes)]

        for i, name in enumerate(class_names):
            col_name = name.replace(" ", "_") + "_prob"
            data[col_name] = results['probabilities'][:, i]

        results_df = pd.DataFrame(data)

    results_df.to_csv(output_path, index=False)
    print(f"Results saved to: {output_path}")
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run inference on bioacoustic sounds")

    # Config file (optional)
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file")

    # Audio source
    parser.add_argument("--audios_source", type=str, required=False,
                        help="Path to folder, JSON, or CSV with windows")

    # Cascade mode: pre-computed spectrograms folder + two checkpoints
    parser.add_argument("--spectrograms_dir", type=str, default=None,
                        help="Path to folder of pre-computed .npy spectrograms (enables cascade mode)")
    parser.add_argument("--checkpoint_binary", type=str, default=None,
                        help="Binary model checkpoint (cascade mode)")
    parser.add_argument("--checkpoint_3class", type=str, default=None,
                        help="3-class model checkpoint (cascade mode)")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Output CSV path for cascade results")
    parser.add_argument("--target_size", type=int, nargs=2, default=None,
                        metavar=("H", "W"),
                        help="Override spectrogram target size (H W), e.g. --target_size 224 180")

    # Classification mode
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes (2=binary, >2=multiclass)")
    parser.add_argument("--class_names", type=str, nargs="+", default=None,
                        help="Class names for multiclass")

    # Audio parameters
    parser.add_argument("--window_size_sec", type=float, default=5.0)
    parser.add_argument("--overlap_sec", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=48000)

    # Spectrogram parameters
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--hop_length", type=int, default=512)
    parser.add_argument("--n_mels", type=int, default=224)
    parser.add_argument("--top_db", type=float, default=80.0)

    # Model and inference
    parser.add_argument("--checkpoint", type=str, required=False)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--conf_threshold", type=float, default=0.5,
                        help="Probability threshold for the binary classifier (default 0.5)")

    # Output
    parser.add_argument("--dataset", type=str, help="Dataset name for output directory")
    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--spectrograms_path", type=str, default=None)
    parser.add_argument("--annotations_json", type=str, default=None,
                        help="Annotations JSON for mapping sound IDs to paths")

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    #  Cascade mode: spectrograms_dir + checkpoint_binary + checkpoint_3class
    # ------------------------------------------------------------------ #
    if args.spectrograms_dir is not None:
        if not args.checkpoint_binary or not args.checkpoint_3class:
            parser.error("--spectrograms_dir requires both --checkpoint_binary and --checkpoint_3class")

        if args.device == "cuda" and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU")
            args.device = "cpu"
        print(f"Using device: {args.device}")

        # Build DataFrame from spectrogram filenames
        print(f"Scanning spectrograms in: {args.spectrograms_dir}")
        meta_df = build_dataframe_from_spectrograms_dir(args.spectrograms_dir, args.sample_rate)
        print(f"Found {len(meta_df)} spectrograms")

        if args.target_size is not None:
            target_size = tuple(args.target_size)
        else:
            n_frames = int(np.ceil((args.window_size_sec * args.sample_rate - args.n_fft) / args.hop_length)) + 1
            target_size = (args.n_mels, n_frames)
        print(f"Spectrogram target size: {target_size}")

        dataset = BioacousticsInferenceDataset(
            dataframe=meta_df,
            x_col='file_path',
            target_size=target_size,
            normalize=args.normalize,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=(args.device == "cuda"),
        )

        # --- Binary model ---
        print("\nLoading binary model...")
        binary_model = load_model_from_checkpoint(args.checkpoint_binary, args.device)
        print("Running binary inference...")
        binary_results = run_inference_batch(
            model=binary_model,
            dataloader=dataloader,
            sample_rate=args.sample_rate,
            num_classes=2,
            device=args.device,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
            meta_df=meta_df,
        )

        # --- 3-class model ---
        print("\nLoading 3-class model...")
        class3_model = load_model_from_checkpoint(args.checkpoint_3class, args.device)
        print("Running 3-class inference...")
        class3_results = run_inference_batch(
            model=class3_model,
            dataloader=dataloader,
            sample_rate=args.sample_rate,
            num_classes=3,
            device=args.device,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
            meta_df=meta_df,
        )

        # --- Merge results ---
        out_df = meta_df.copy()
        out_df['pred_label_binary'] = binary_results['predictions']
        out_df['prob_class_0'] = 1 - binary_results['probabilities']
        out_df['confidence_binary'] = np.abs(binary_results['probabilities'] - 0.5) * 2
        out_df['pred_label_3class'] = class3_results['predictions'] + 1  # remap 0,1,2 → 1,2,3
        out_df['prob_class_1'] = class3_results['probabilities'][:, 0]
        out_df['prob_class_2'] = class3_results['probabilities'][:, 1]
        out_df['prob_class_3'] = class3_results['probabilities'][:, 2]
        out_df['pred_label'] = out_df.apply(
            lambda r: 0 if r['pred_label_binary'] == 0 else int(r['pred_label_3class']),
            axis=1,
        )

        # Reorder columns
        out_df = out_df[[
            'file_path', 'audio', 'start(s)', 'end(s)',
            'pred_label', 'pred_label_binary', 'prob_class_0', 'confidence_binary',
            'pred_label_3class', 'prob_class_1', 'prob_class_2', 'prob_class_3',
        ]]

        # Save
        if args.output_csv:
            output_path = args.output_csv
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        else:
            dataset_name = args.dataset or Path(args.spectrograms_dir).name
            output_dir = os.path.join(".", "inference", dataset_name)
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, "cascade_results.csv")

        out_df.to_csv(output_path, index=False)
        print(f"\nCascade results saved to: {output_path}")
        print("Inference pipeline completed successfully!")
        return

    # Load config if provided
    if args.config:
        cfg = load_config(args.config)

        if args.num_classes == 2 and cfg.training.num_classes != 2:
            args.num_classes = cfg.training.num_classes
        if args.class_names is None and cfg.class_names:
            args.class_names = list(cfg.class_names.values())
        if args.window_size_sec == 5.0:
            args.window_size_sec = cfg.audio.window_size_sec
        if args.overlap_sec == 4.0:
            args.overlap_sec = cfg.audio.overlap_sec
        if args.sample_rate == 48000:
            args.sample_rate = cfg.audio.sample_rate
        if args.hop_length == 512:
            args.hop_length = cfg.spectrogram.hop_length
        if args.n_mels == 224:
            args.n_mels = cfg.spectrogram.n_mels
        if args.n_fft == 2048:
            args.n_fft = cfg.spectrogram.n_fft
        if args.top_db == 80.0:
            args.top_db = cfg.spectrogram.top_db
        if not args.dataset:
            args.dataset = cfg.name

    is_binary = (args.num_classes == 2)
    print(f"Running {'binary' if is_binary else f'multiclass ({args.num_classes} classes)'} inference")

    # Build windows
    if args.audios_source.endswith('.json'):
        with open(args.audios_source, 'r') as in_file:
            windows = json.load(in_file)
        df = pd.DataFrame(windows)
    elif args.audios_source.endswith('.csv'):
        df = pd.read_csv(args.audios_source)
        windows = df.to_dict('records')
    else:
        windows = build_inference_windows(
            audios_source=args.audios_source,
            window_size_sec=args.window_size_sec,
            overlap_sec=args.overlap_sec,
            sample_rate=args.sample_rate,
        )
        df = pd.DataFrame(windows)
        output_dir = os.path.join(".", "inference", args.dataset)
        os.makedirs(output_dir, exist_ok=True)
        windows_path = os.path.join(output_dir, f"{args.dataset}_windows.json")
        with open(windows_path, 'w') as out_file:
            json.dump(windows, out_file, indent=2)
        print(f"Windows saved to: {windows_path}")

    # Setup output and spectrograms directories
    output_dir = os.path.join(".", "inference", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    if args.spectrograms_path:
        spectrograms_path = args.spectrograms_path
    else:
        spectrograms_path = os.path.join(output_dir, "spectrograms")
        os.makedirs(spectrograms_path, exist_ok=True)
        compute_mel_spectrograms_gpu(
            windows=windows,
            sample_rate=args.sample_rate,
            n_fft=args.n_fft,
            hop_length=args.hop_length,
            n_mels=args.n_mels,
            top_db=args.top_db,
            spectrograms_path=spectrograms_path,
            save_npy=True,
            fill_highfreq=True,
            noise_db_mean=None,
            noise_db_std=3.0,
            storage_dtype="float32",
        )

    # Build spec_name column — try both standard and legacy naming conventions
    if 'spec_name' not in df.columns and 'file_path' not in df.columns:
        df['spec_name'] = df.apply(
            lambda row: resolve_spectrogram_path(
                spectrograms_path,
                sound_path=row.get('sound_path'),
                start=int(row['start']),
                end=int(row['end']),
                sound_id=row.get('sound_id'),
                window_id=row.get('window_id'),
                label=row.get('label'),
            ),
            axis=1,
        )

    x_col = 'file_path' if 'file_path' in df.columns else 'spec_name'

    # Calculate target size
    n_frames = int(np.ceil((args.window_size_sec * args.sample_rate - args.n_fft) / args.hop_length)) + 1
    target_size = (args.n_mels, n_frames)
    print(f"Spectrogram size: {target_size}")

    # Create dataset
    dataset = BioacousticsInferenceDataset(
        dataframe=df,
        x_col=x_col,
        target_size=target_size,
        normalize=args.normalize,
    )
    print(f"Created dataset with {len(dataset)} samples")

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if args.device == "cuda" else False
    )

    # Check device availability
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, switching to CPU")
        args.device = "cpu"
    print(f"Using device: {args.device}")

    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, args.device)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Run inference
    try:
        results = run_inference_batch(
            model=model,
            dataloader=dataloader,
            sample_rate=args.sample_rate,
            num_classes=args.num_classes,
            annotations_json=args.annotations_json,
            device=args.device,
            temperature=args.temperature,
            conf_threshold=args.conf_threshold,
        )
        print("Inference completed successfully")
    except Exception as e:
        print(f"Error during inference: {e}")
        return

    # Save results
    suffix = "binary" if is_binary else "multiclass"
    results_path = os.path.join(output_dir, f"{suffix}_inference_results.csv")
    save_inference_results(
        results=results,
        output_path=results_path,
        num_classes=args.num_classes,
        class_names=args.class_names,
    )

    print("Inference pipeline completed successfully!")


if __name__ == "__main__":
    main()