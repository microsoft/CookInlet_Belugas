"""
Dataset preparation script for Cook Inlet Belugas bioacoustics project.

Usage:
    # Full pipeline
    python prepare_dataset.py --config data/data_config.yaml

    # Run specific steps only
    python prepare_dataset.py --config data/data_config.yaml --steps stats windows

    # Available steps: stats, plot, windows, spectrograms, splits
    # - stats: Display dataset statistics
    # - plot: Create distribution plots (by dataset and splits)
    # - windows: Build windows from annotations
    # - spectrograms: Compute mel spectrograms
    # - splits: Create train/val/test splits
"""

import os
import argparse
import json
from pathlib import Path
from typing import List, Optional

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Import from PytorchWildlife core library
from PytorchWildlife.data.bioacoustics.bioacoustics_configs import load_config, DomainConfig
from PytorchWildlife.data.bioacoustics.bioacoustics_windows import build_windows, count_window_labels


def run_plot_distribution(config: DomainConfig) -> None:
    """Create pie charts showing class distribution by dataset."""
    print(f"\n{'='*60}")
    print(f"Step: Plot Class Distribution")
    print(f"{'='*60}")

    # Load windows
    windows_path = os.path.join(config.paths.data_root, config.paths.windows_json)
    if not os.path.exists(windows_path):
        print(f"Error: Windows file not found: {windows_path}")
        print("Run 'windows' step first.")
        return

    with open(windows_path, 'r') as f:
        windows = json.load(f)

    print(f"Loaded {len(windows)} windows")

    # Convert to DataFrame for easier grouping
    df = pd.DataFrame(windows)

    # Get datasets
    datasets = sorted(df['dataset'].unique())
    print(f"Datasets: {datasets}")

    # Load class names from config
    class_names = config.class_names

    # Count total distribution
    total_counts = df['label'].value_counts().sort_index().to_dict()

    # Count per dataset
    dataset_counts = {}
    for dataset in datasets:
        dataset_df = df[df['dataset'] == dataset]
        counts = dataset_df['label'].value_counts().sort_index().to_dict()
        dataset_counts[dataset] = counts

    # Print statistics
    print(f"\nTotal distribution (n={len(df)}):")
    for label, count in sorted(total_counts.items()):
        class_name = class_names.get(label, f"Class {label}")
        pct = 100 * count / len(df)
        print(f"  - {class_name} (label {label}): {count} ({pct:.1f}%)")

    for dataset in datasets:
        counts = dataset_counts[dataset]
        dataset_total = sum(counts.values())
        print(f"\n{dataset} distribution (n={dataset_total}):")
        for label, count in sorted(counts.items()):
            class_name = class_names.get(label, f"Class {label}")
            pct = 100 * count / dataset_total
            print(f"  - {class_name} (label {label}): {count} ({pct:.1f}%)")

    # Create visualization
    n_plots = len(datasets) + 1  # Total + per dataset
    fig, axes = plt.subplots(1, n_plots, figsize=(5*n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    # Define colors for each class (consistent across all plots)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    def make_pie_chart(ax, counts, title, total):
        """Helper to create a single pie chart."""
        labels = []
        sizes = []
        chart_colors = []
        
        for label in sorted(counts.keys()):
            count = counts[label]
            class_name = class_names.get(label, f"Class {label}")
            pct = 100 * count / total
            labels.append(f"{pct:.1f}% ({count})")
            sizes.append(count)
            chart_colors.append(colors[label % len(colors)])
        
        wedges, texts, autotexts = ax.pie(
            sizes, 
            labels=labels,
            colors=chart_colors,
            autopct='',
            startangle=90,
            textprops={'fontsize': 10}
        )
        
        ax.set_title(f"{title}\n(n={total})", fontsize=12, fontweight='bold')

    # Plot 1: Total distribution
    make_pie_chart(axes[0], total_counts, "Total", len(df))

    # Plot 2+: Per dataset
    for idx, dataset in enumerate(datasets):
        dataset_total = sum(dataset_counts[dataset].values())
        make_pie_chart(axes[idx+1], dataset_counts[dataset], dataset, dataset_total)

    # Add legend
    legend_elements = []
    for label in sorted(set(df['label'])):
        class_name = class_names.get(label, f"Class {label}")
        legend_elements.append(
            mpatches.Patch(color=colors[label % len(colors)], label=class_name)
        )
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(legend_elements),
        fontsize=11,
        frameon=True
    )

    plt.suptitle("Class distribution: total and by dataset", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.05, 1, 0.96])

    # Save plot
    output_dir = config.paths.data_root
    os.makedirs(os.path.join(output_dir, "dataset_plots"), exist_ok=True)
    plot_path = os.path.join(output_dir, "dataset_plots", "class_distribution_by_dataset.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")
    
    # Also show if in interactive mode
    plt.show()


def run_plot_splits(config: DomainConfig) -> None:
    """Create pie charts showing class distribution for train/val/test splits."""
    print(f"\n{'='*60}")
    print(f"Step: Plot Splits Distribution (4-class)")
    print(f"{'='*60}")

    # Check for splits directory
    splits_dir = os.path.join(config.paths.data_root, "splits_4class")
    if not os.path.exists(splits_dir):
        print(f"Error: Splits directory not found: {splits_dir}")
        print("Run 'splits' step first.")
        return

    # Load splits
    split_files = {
        'Train': os.path.join(splits_dir, "train_split.csv"),
        'Val': os.path.join(splits_dir, "val_split.csv"),
        'Test': os.path.join(splits_dir, "test_split.csv")
    }

    # Check all files exist
    for split_name, filepath in split_files.items():
        if not os.path.exists(filepath):
            print(f"Error: Split file not found: {filepath}")
            return

    # Load data
    splits_data = {}
    for split_name, filepath in split_files.items():
        df = pd.read_csv(filepath)
        splits_data[split_name] = df
        print(f"Loaded {split_name}: {len(df)} samples")

    # Load class names from config
    class_names = config.class_names

    # Define colors for each class (consistent with dataset plot)
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']  # Blue, Orange, Green, Red
    
    # Create figure with 3 subplots (Train, Val, Test)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    def make_split_pie_chart(ax, df, title):
        """Helper to create a single pie chart for a split."""
        counts = df['label'].value_counts().sort_index().to_dict()
        total = len(df)
        
        labels = []
        sizes = []
        chart_colors = []
        
        for label in sorted(counts.keys()):
            count = counts[label]
            pct = 100 * count / total
            
            # Create label text with percentage and count
            labels.append(f"{pct:.1f}%\n({count})")
            sizes.append(count)
            chart_colors.append(colors[label % len(colors)])
        
        # Create pie chart
        wedges, texts = ax.pie(
            sizes, 
            labels=labels,
            colors=chart_colors,
            startangle=90,
            textprops={'fontsize': 9}
        )
        
        # Add title with total count
        ax.set_title(f"{title} (n={total})", fontsize=14, fontweight='bold')

    # Plot each split
    for idx, split_name in enumerate(['Train', 'Val', 'Test']):
        make_split_pie_chart(axes[idx], splits_data[split_name], split_name)

    # Add legend with class names
    legend_elements = []
    # Get all unique labels across all splits
    all_labels = set()
    for df in splits_data.values():
        all_labels.update(df['label'].unique())
    
    for label in sorted(all_labels):
        class_name = class_names.get(label, f"Class {label}")
        legend_elements.append(
            mpatches.Patch(color=colors[label % len(colors)], label=f"{label} ({class_name})")
        )
    
    fig.legend(
        handles=legend_elements,
        loc='lower center',
        ncol=len(legend_elements),
        fontsize=11,
        frameon=True
    )

    # Add overall title
    plt.suptitle("4-class splits distribution", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0.08, 1, 0.95])

    # Save plot
    output_dir = os.path.join(config.paths.data_root, "dataset_plots")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, "splits_4class_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot to: {plot_path}")

    # Print statistics
    for split_name in ['Train', 'Val', 'Test']:
        df = splits_data[split_name]
        counts = df['label'].value_counts().sort_index().to_dict()
        print(f"\n{split_name} distribution (n={len(df)}):")
        for label in sorted(counts.keys()):
            count = counts[label]
            class_name = class_names.get(label, f"Class {label}")
            pct = 100 * count / len(df)
            print(f"  - {label} ({class_name}): {count} ({pct:.1f}%)")
    
    # Also show if in interactive mode
    plt.show()


def run_stats(config: DomainConfig) -> None:
    """Load and display dataset statistics."""
    print(f"\n{'='*60}")
    print(f"Step: Dataset Statistics")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    print(f"Loading annotations from: {annotation_path}")

    if not os.path.exists(annotation_path):
        print(f"Warning: Annotations file not found: {annotation_path}")
        return

    with open(annotation_path, 'r') as f:
        data = json.load(f)

    # Dataset info
    if 'info' in data:
        print(f"\nDataset Info:")
        for key, value in data['info'].items():
            print(f"  - {key}: {value}")

    # Sound statistics
    sounds = data.get('sounds', [])
    print(f"\nSounds: {len(sounds)}")
    if sounds:
        durations = [s.get('duration', 0) for s in sounds]
        print(f"  - Total duration: {sum(durations):.1f}s ({sum(durations)/3600:.2f}h)")
        print(f"  - Mean duration: {sum(durations)/len(durations):.1f}s")
        print(f"  - Min duration: {min(durations):.1f}s")
        print(f"  - Max duration: {max(durations):.1f}s")

    # Annotation statistics
    annotations = data.get('annotations', [])
    print(f"\nAnnotations: {len(annotations)}")
    if annotations:
        categories = {}
        for ann in annotations:
            cat_id = ann.get('category_id', 0)
            categories[cat_id] = categories.get(cat_id, 0) + 1
        print(f"  - By category: {categories}")

    # Category names
    if 'categories' in data:
        print(f"\nCategories:")
        for cat in data['categories']:
            print(f"  - {cat.get('id', '?')}: {cat.get('name', 'Unknown')}")


def run_windows(config: DomainConfig) -> List[dict]:
    """Build windows from annotations."""
    print(f"\n{'='*60}")
    print(f"Step: Build Windows")
    print(f"{'='*60}")

    annotation_path = config.paths.annotations_path
    output_dir = config.paths.data_root
    os.makedirs(output_dir, exist_ok=True)

    windows_output_path = os.path.join(
        output_dir,
        config.paths.windows_json,
    )

    if os.path.exists(windows_output_path):
        print(f"Loading existing windows from: {windows_output_path}")
        with open(windows_output_path, 'r') as f:
            windows = json.load(f)
        print(f"Loaded {len(windows)} windows")
    else:
        strategy = config.audio.window_strategy
        print(f"Building windows with:")
        print(f"  - strategy: {strategy}")
        print(f"  - window_size: {config.audio.window_size_sec}s")
        print(f"  - overlap: {config.audio.overlap_sec}s")
        print(f"  - sample_rate: {config.audio.sample_rate}")
        print(f"  - datasets: {config.datasets}")
        if strategy == "balanced":
            print(f"  - negative_proportion: {config.audio.negative_proportion}")
        print(f"  - min_overlap_sec: {config.audio.min_overlap_sec}")

        windows = build_windows(
            annotation_file=annotation_path,
            window_size_sec=config.audio.window_size_sec,
            overlap_sec=config.audio.overlap_sec,
            sample_rate=config.audio.sample_rate,
            datasets_names=config.datasets,
            strategy=strategy,
            negative_proportion=config.audio.negative_proportion,
            multiclass=config.audio.multiclass,
            min_overlap_sec=config.audio.min_overlap_sec,
        )

        with open(windows_output_path, 'w') as f:
            json.dump(windows, f, indent=2)
        print(f"Saved {len(windows)} windows to: {windows_output_path}")

    # Show label distribution
    counts = count_window_labels(windows)
    print(f"\nLabel distribution: {counts}")

    return windows


def run_spectrograms(config: DomainConfig, windows: List[dict]) -> None:
    """Compute mel spectrograms using GPU."""
    # Import here to avoid loading torch unnecessarily
    from PytorchWildlife.data.bioacoustics.bioacoustics_spectrograms import compute_mel_spectrograms_gpu
    from inference import resolve_spectrogram_path

    print(f"\n{'='*60}")
    print(f"Step: Compute Mel Spectrograms (GPU)")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    os.makedirs(spectrograms_dir, exist_ok=True)

    print(f"Output directory: {spectrograms_dir}")
    print(f"Spectrogram parameters:")
    print(f"  - n_fft: {config.spectrogram.n_fft}")
    print(f"  - hop_length: {config.spectrogram.hop_length}")
    print(f"  - n_mels: {config.spectrogram.n_mels}")
    print(f"  - top_db: {config.spectrogram.top_db}")
    print(f"  - fill_highfreq: {config.spectrogram.fill_highfreq}")

    # Load annotations to get audio file paths
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)

    sounds = {s['id']: s for s in annotations['sounds']}

    # Convert windows format to include sound_path (keep legacy keys for lookup)
    inference_windows = []
    for win in windows:
        sound = sounds.get(win['sound_id'])
        if sound:
            # Resolve audio file path relative to data_root
            audio_path = sound['file_name_path']
            if not os.path.isabs(audio_path):
                audio_path = os.path.join(config.paths.data_root, audio_path)
            
            inference_windows.append({
                'window_id': win['window_id'],
                'sound_id': win['sound_id'],
                'sound_path': audio_path,
                'start': win['start'],
                'end': win['end'],
                'label': win.get('label'),
            })

    def _legacy_spectrogram_path(win, spectrograms_path):
        return resolve_spectrogram_path(
            spectrograms_path,
            sound_path=win.get("sound_path"),
            start=int(win["start"]),
            end=int(win["end"]),
            sound_id=win.get("sound_id"),
            window_id=win.get("window_id"),
            label=win.get("label"),
        )

    compute_mel_spectrograms_gpu(
        windows=inference_windows,
        sample_rate=config.audio.sample_rate,
        n_fft=config.spectrogram.n_fft,
        hop_length=config.spectrogram.hop_length,
        n_mels=config.spectrogram.n_mels,
        top_db=config.spectrogram.top_db,
        spectrograms_path=spectrograms_dir,
        save_npy=True,
        fill_highfreq=config.spectrogram.fill_highfreq,
        noise_db_std=config.spectrogram.noise_db_std,
        storage_dtype=config.spectrogram.storage_dtype,
        spectrogram_path_fn=_legacy_spectrogram_path,
    )

    print("Spectrogram computation complete!")


def run_splits(config: DomainConfig, windows: List[dict]) -> None:
    """Create grouped train/val/test splits, with optional binary conversion.

    Steps:
        1. Build a DataFrame from the windows mapping.
        2. Generate the spectrogram file name expected on disk and filter out
           windows whose spectrogram has not been computed yet.
        3. Split into train+val / test (grouped by ``sound_id``), then
           train / val (stratified + grouped).
        4. Save CSVs under ``<data_root>/splits/``.
    """
    from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

    print(f"\n{'='*60}")
    print(f"Step: Create Data Splits")
    print(f"{'='*60}")

    spectrograms_dir = config.paths.spectrograms_dir
    output_dir = config.paths.data_root

    print(f"Spectrograms directory: {spectrograms_dir}")
    print(f"Splits output directory: {output_dir}")
    print(f"Split parameters:")
    print(f"  - test_size: {config.splits.test_size}")
    print(f"  - val_size: {config.splits.val_size}")
    print(f"  - random_state: {config.splits.random_state}")

    # --- Build DataFrame from windows ---
    df = pd.DataFrame(windows)

    # Load annotations to map sound_id -> file path
    with open(config.paths.annotations_path, 'r') as f:
        annotations = json.load(f)
    sounds = {s['id']: s for s in annotations['sounds']}

    # Build spectrogram file name column
    # Try standard naming first; fall back to legacy naming per row
    df['sound_filename'] = df['sound_id'].map(
        lambda sid: os.path.splitext(
            os.path.basename(sounds[sid]['file_name_path'])
        )[0]
    )

    def _resolve_spec_name(row):
        """Return the spectrogram filename that exists on disk."""
        standard = f"{row['sound_filename']}_{row['start']}_{row['end']}.npy"
        if os.path.exists(os.path.join(spectrograms_dir, standard)):
            return standard
        legacy = (
            f"sid{row['sound_id']}_idx{row['window_id']}"
            f"_start{row['start']}_end{row['end']}_lab{row['label']}.npy"
        )
        if os.path.exists(os.path.join(spectrograms_dir, legacy)):
            return legacy
        # Default to standard (will be filtered out later)
        return standard

    df['spec_name'] = df.apply(_resolve_spec_name, axis=1)

    # Filter to rows where the spectrogram .npy exists on disk
    df['spec_exists'] = df['spec_name'].apply(
        lambda x: os.path.exists(os.path.join(spectrograms_dir, x))
    )
    print(f"\nTotal windows: {len(df)}")
    print(f"Existing spectrograms: {df['spec_exists'].sum()}")
    df = df[df['spec_exists']].drop(columns=['spec_exists'])

    if len(df) == 0:
        print("Error: No spectrograms found. Run 'spectrograms' step first.")
        return

    # --- Train+Val / Test split (grouped by sound_id) ---
    gss = GroupShuffleSplit(
        n_splits=1,
        test_size=config.splits.test_size,
        random_state=config.splits.random_state,
    )
    trainval_idx, test_idx = next(
        gss.split(df, df['label'], groups=df['sound_id'])
    )
    trainval_df = df.iloc[trainval_idx].copy()
    test_df = df.iloc[test_idx].copy()

    # --- Train / Val split (stratified + grouped by sound_id) ---
    sgkf = StratifiedGroupKFold(
        n_splits=config.splits.n_splits, shuffle=True,
        random_state=config.splits.random_state,
    )
    train_idx, val_idx = next(
        sgkf.split(trainval_df, trainval_df['label'], trainval_df['sound_id'])
    )
    train_df = trainval_df.iloc[train_idx].copy()
    val_df = trainval_df.iloc[val_idx].copy()

    # --- Save 4-class splits ---
    splits_dir = os.path.join(output_dir, "splits_4class")
    os.makedirs(splits_dir, exist_ok=True)

    train_df.to_csv(os.path.join(splits_dir, "train_split.csv"), index=False)
    val_df.to_csv(os.path.join(splits_dir, "val_split.csv"), index=False)
    test_df.to_csv(os.path.join(splits_dir, "test_split.csv"), index=False)

    for name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
        label_counts = split_df['label'].value_counts().to_dict()
        print(f"  {name:5s} (n={len(split_df)}): {label_counts}")

    print(f"Saved splits to: {splits_dir}")

    # --- Derived splits ---
    # --- 3-class splits: drop noise, remap {1→0, 2→1, 3→2} ---
    three_class_dir = os.path.join(output_dir, "splits_3class")
    os.makedirs(three_class_dir, exist_ok=True)
    remap_3class = {1: 0, 2: 1, 3: 2}
    print(f"\nCreating 3-class splits (drop class 0, remap {remap_3class})")

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        three_df = _remap_labels(
            split_df, label_map=remap_3class, drop_unmapped=True
        )
        three_df.to_csv(
            os.path.join(three_class_dir, f"{name}_split.csv"), index=False
        )
        label_counts = three_df['label'].value_counts().to_dict()
        print(f"  {name.capitalize():5s} 3-class (n={len(three_df)}): {label_counts}")

    print(f"Saved 3-class splits to: {three_class_dir}")

    # --- Binary splits: all positive classes → 1 ---
    binary_dir = os.path.join(output_dir, "splits_binary")
    os.makedirs(binary_dir, exist_ok=True)

    positive_classes = [c for c in sorted(df['label'].unique()) if c != 0]
    print(f"\nCreating binary splits (positive classes: {positive_classes})")

    for name, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        binary_df = _convert_to_binary(split_df, positive_classes)
        binary_df.to_csv(
            os.path.join(binary_dir, f"{name}_split.csv"), index=False
        )
        label_counts = binary_df['label'].value_counts().to_dict()
        print(f"  {name.capitalize():5s} binary (n={len(binary_df)}): {label_counts}")

    print(f"Saved binary splits to: {binary_dir}")


def _convert_to_binary(
    df: pd.DataFrame,
    positive_classes: List[int],
    label_col: str = "label",
) -> pd.DataFrame:
    """Return a copy of *df* with *label_col* mapped to binary (0/1).

    Any label in *positive_classes* becomes 1; everything else becomes 0.
    """
    binary_df = df.copy()
    binary_df[label_col] = binary_df[label_col].apply(
        lambda x: 1 if x in positive_classes else 0
    )
    return binary_df


def _remap_labels(
    df: pd.DataFrame,
    label_map: dict,
    label_col: str = "label",
    drop_unmapped: bool = True,
) -> pd.DataFrame:
    """Return a copy of *df* with labels remapped according to *label_map*.

    Args:
        df: Input DataFrame.
        label_map: Mapping ``{old_label: new_label}``.
        label_col: Column containing the labels.
        drop_unmapped: If True, rows whose label is not in *label_map* are
            dropped.  Otherwise they are kept unchanged.
    """
    out = df.copy()
    if drop_unmapped:
        out = out[out[label_col].isin(label_map)]
    out[label_col] = out[label_col].map(label_map).fillna(out[label_col]).astype(int)
    return out


def load_windows_if_exists(config: DomainConfig) -> Optional[List[dict]]:
    """Load windows from file if they exist."""
    output_dir = config.paths.data_root
    windows_output_path = os.path.join(
        output_dir,
        config.paths.windows_json,
    )

    if os.path.exists(windows_output_path):
        with open(windows_output_path, 'r') as f:
            return json.load(f)
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Prepare dataset for training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to YAML config file (e.g., config/template.yaml)"
    )
    parser.add_argument(
        "--steps", type=str, nargs="+",
        default=["stats", "windows", "spectrograms", "splits"],
        choices=["stats", "plot", "windows", "spectrograms", "splits"],
        help="Steps to run (default: all)"
    )

    args = parser.parse_args()

    # Load configuration
    print(f"Loading config from: {args.config}")
    config = load_config(args.config)

    # Track windows (needed for some steps)
    windows = None

    # Run requested steps
    if "stats" in args.steps:
        run_stats(config)

    if "plot" in args.steps:
        run_plot_distribution(config)
        run_plot_splits(config)

    if "windows" in args.steps:
        windows = run_windows(config)
    elif "spectrograms" in args.steps or "splits" in args.steps:
        windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return

    if "spectrograms" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_spectrograms(config, windows)

    if "splits" in args.steps:
        if windows is None:
            windows = load_windows_if_exists(config)
        if windows is None:
            print("\nError: Windows not found. Run 'windows' step first.")
            return
        run_splits(config, windows)

    print(f"\n{'='*60}")
    print("Dataset preparation complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
