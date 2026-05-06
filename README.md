# Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes

This repository contains the code and experiment pipelines required to reproduce the results presented in: Castellote et al. 2026. Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes. Marine Mammal Science.

### Overview

This work is part of the long-term Cook Inlet Beluga Acoustics (CIBA) Program, a collaborative effort led by the Marine Mammal Laboratory, Alaska Fisheries Science Center (NOAA) in partnership with the Division of Wildlife Conservation, Alaska Department of Fish and Game. This framework supersedes the previous AI pipeline developed by Ming et al. (2019): https://github.com/Microsoft/belugasounds.

### Key Features

The repository implements an end-to-end workflow for analyzing archival sound recordings from moored instruments in Cook Inlet, Alaska, featuring:
- **Automated Processing:** End-to-end spectrogram generation from raw audio.
- **Two-Stage Deep Learning:** A specialized architecture for cetacean signal detection and multi-species classification (Beluga, Humpback, and Killer Whale).
- **Domain Adaptation:** An active-learning loop designed to adapt the model to new soundscapes as monitoring expands into unsampled geographic areas.
- **Comprehensive Toolset:** Modules to support model training, evaluation, and large-scale inference on long-duration recordings.

The codebase is organized to facilitate the full research lifecycle, from initial training to the replication of the final framework proposed in the manuscript.

![Cook Inlet beluga whales](belugas.JPG)
Photo: M. Castellote, beluga whales off Fire Island, Cook Inlet, Alaska.

## Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Audio data and annotations are available upon request to the corresponding author. Place all audio files and the annotations JSON under the `data/` directory.

The preparation pipeline is controlled by a single YAML config file (`data/data_config.yaml`) that defines audio parameters, spectrogram settings, and split ratios. It runs four steps in order:

| Step | Description |
|------|-------------|
| `stats` | Print dataset statistics (number of sounds, total duration, annotation counts per category). |
| `windows` | Slide fixed-size windows over each audio file and assign labels based on annotation overlap. Saves a JSON mapping. |
| `spectrograms` | Compute GPU-accelerated mel spectrograms for every window and save them as `.npy` files. |
| `splits` | Create grouped train/val/test splits (stratified by label, grouped by sound). It generates `splits_4class/`, `splits_3class/` (positive classes only, remapped), and `splits_binary/` (all positive classes mapped to 1) variants. |

```bash
# Run the full pipeline
python prepare_dataset.py --config data/data_config.yaml

# Run specific steps only (e.g., statistics and windows)
python prepare_dataset.py --config data/data_config.yaml --steps stats windows

# Compute spectrograms only (windows must already exist)
python prepare_dataset.py --config data/data_config.yaml --steps spectrograms

# Create splits only (windows and spectrograms must already exist)
python prepare_dataset.py --config data/data_config.yaml --steps splits
```

After running the full pipeline the `data/` directory will contain:

```
data/
├── data_config.yaml
├── annotations.json
├── windows.json
├── mel_spectrograms_multiclass/   # .npy spectrograms
├── splits_4class/                 # 4-class CSVs (noise + 3 species, ~75% no-whale)
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── splits_4class_25/              # 4-class CSVs (no-whale downsampled to 25%)
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── splits_3class/                 # 3-class CSVs (species only)
│   └── ...
└── splits_binary/                 # Binary CSVs (whale vs. no-whale)
    └── ...
```

### 3. Train Base Models

All model and training parameters are defined in YAML configs under `configs/`. Only runtime arguments (split paths, checkpoint path) are passed via CLI.

```bash
# Binary classification — Stage 1 (whale / no-whale), ResNet18
python train.py --config configs/config_binary.yaml \
    --train_csv data/splits_binary/train_split.csv \
    --val_csv data/splits_binary/val_split.csv \
    --test_csv data/splits_binary/test_split.csv
```

```bash
# 3-class species classification — Stage 2 (humpback / orca / beluga), ResNet34
python train.py --config configs/config_3class.yaml \
    --train_csv data/splits_3class/train_split.csv \
    --val_csv data/splits_3class/val_split.csv \
    --test_csv data/splits_3class/test_split.csv
```

```bash
# 4-class - Stage 1 (no-whale + 3 species), ResNet34
python train.py --config configs/config_4class_75.yaml \
    --train_csv data/splits_4class/train_split.csv \
    --val_csv data/splits_4class/val_split.csv \
    --test_csv data/splits_4class/test_split.csv
```

```bash
# 4-class — Stage 2 (no-whale + 3 species), ResNet34
python train.py --config configs/config_4class_25.yaml \
    --train_csv data/splits_4class_25/train_split.csv \
    --val_csv data/splits_4class_25/val_split.csv \
    --test_csv data/splits_4class_25/test_split.csv
```

### 4. Test Base Models on New Deployment Sites

To evaluate the base models on a new deployment site, run each stage separately and then combine their predictions using `compare_models.py`. The binary model detects whale presence and the 3-class model identifies the species; `compare_models.py` applies the cascade logic and computes final metrics. The `--predict_only` flag skips metric computation and only exports a predictions CSV, which is useful when the test label space differs from the model's label space (e.g., a site with only beluga annotations).

#### Tuxedni Channel

```bash
python train.py --config configs/config_binary.yaml \
    --test_csv data/tuxedni_splits/final_test.csv \
    --ckpt_path checkpoints/binary/best.ckpt \
    --spectrograms_dir data/tuxedni_final_test_spectrograms \
    --exp_name tuxedni_final_test \
    --output_csv binary.csv \
    --predict_only
```

```bash
python train.py --config configs/config_3class.yaml \
    --test_csv data/tuxedni_splits/final_test.csv \
    --ckpt_path checkpoints/3class/best.ckpt \
    --spectrograms_dir data/tuxedni_final_test_spectrograms \
    --exp_name tuxedni_final_test \
    --output_csv 3class.csv \
    --predict_only
```

```bash
python compare_models.py --binary_3class_only \
    --pred_binary test_results/tuxedni_final_test/binary.csv \
    --pred_3class test_results/tuxedni_final_test/3class.csv
```

#### Johnson River

```bash
python train.py --config configs/config_binary.yaml \
    --test_csv data/johnson_splits/final_test.csv \
    --ckpt_path checkpoints/binary/best.ckpt \
    --spectrograms_dir data/johnson_final_test_spectrograms \
    --exp_name johnson_final_test \
    --output_csv binary.csv \
    --predict_only
```

```bash
python train.py --config configs/config_3class.yaml \
    --test_csv data/johnson_splits/final_test.csv \
    --ckpt_path checkpoints/3class/best.ckpt \
    --spectrograms_dir data/johnson_final_test_spectrograms \
    --exp_name johnson_final_test \
    --output_csv 3class.csv \
    --predict_only
```

```bash
python compare_models.py --binary_3class_only \
    --pred_binary test_results/johnson_final_test/binary.csv \
    --pred_3class test_results/johnson_final_test/3class.csv
```

### 5. Active Learning Training

The active learning loop adapts the base models to new deployment sites using a small set of annotated examples from the target soundscape. These annotations were obtained from the predictions of the base models on each site (Section 4): high-confidence predictions were directly used as labels, while low-confidence predictions were reviewed and manually annotated. Each model is then fine-tuned from its base checkpoint using this site-specific training data.

#### Tuxedni Channel

Fine-tune the binary detector and species classifier on Tuxedni data, then evaluate the adapted cascade:

```bash
python train.py --config configs/tuxedni/binary.yaml \
    --train_csv data/tuxedni_splits/train_binary.csv \
    --val_csv data/tuxedni_splits/val_binary.csv \
    --ckpt_path checkpoints/binary/best.ckpt \
    --finetune
```

```bash
python train.py --config configs/config_binary.yaml \
    --ckpt_path checkpoints/tuxedni_binary-finetune/best.ckpt \
    --test_csv data/tuxedni_splits/final_test.csv \
    --spectrograms_dir data/tuxedni_final_test_spectrograms \
    --exp_name tuxedni_final_test_finetuned \
    --output_csv binary.csv \
    --predict_only
```

```bash
python train.py --config configs/tuxedni/3class.yaml \
    --train_csv data/tuxedni_splits/train_3class.csv \
    --val_csv data/tuxedni_splits/val_3class.csv \
    --ckpt_path checkpoints/3class/best.ckpt \
    --finetune
```

```bash
python train.py --config configs/config_3class.yaml \
    --ckpt_path checkpoints/tuxedni_3class-finetune/best.ckpt \
    --test_csv data/tuxedni_splits/final_test.csv \
    --spectrograms_dir data/tuxedni_final_test_spectrograms \
    --exp_name tuxedni_final_test_finetuned \
    --output_csv 3class.csv \
    --predict_only
```

```bash
python compare_models.py --binary_3class_only \
    --pred_binary test_results/tuxedni_final_test_finetuned/binary.csv \
    --pred_3class test_results/tuxedni_final_test_finetuned/3class.csv
```

#### Johnson River

Same fine-tuning and evaluation workflow applied to the Johnson River deployment:

```bash
python train.py --config configs/johnson/binary.yaml \
    --train_csv data/johnson_splits/train_binary.csv \
    --val_csv data/johnson_splits/val_binary.csv \
    --ckpt_path checkpoints/binary/best.ckpt \
    --finetune
```

```bash
python train.py --config configs/config_binary.yaml \
    --ckpt_path checkpoints/johnson_binary-finetune/best.ckpt \
    --test_csv data/johnson_splits/final_test.csv \
    --spectrograms_dir data/johnson_final_test_spectrograms \
    --exp_name johnson_final_test_finetuned \
    --output_csv binary.csv \
    --predict_only
```

```bash
python train.py --config configs/johnson/3class.yaml \
    --train_csv data/johnson_splits/train_3class.csv \
    --val_csv data/johnson_splits/val_3class.csv \
    --ckpt_path checkpoints/3class/best.ckpt \
    --finetune
```

```bash
python train.py --config configs/config_3class.yaml \
    --ckpt_path checkpoints/johnson_3class-finetune/best.ckpt \
    --test_csv data/johnson_splits/final_test.csv \
    --spectrograms_dir data/johnson_final_test_spectrograms \
    --exp_name johnson_final_test_finetuned \
    --output_csv 3class.csv \
    --predict_only
```

```bash
python compare_models.py --binary_3class_only \
    --pred_binary test_results/johnson_final_test_finetuned/binary.csv \
    --pred_3class test_results/johnson_final_test_finetuned/3class.csv
```

### 6. Inference on Unannotated Recordings

Once the models have been fine-tuned to a specific deployment site (Section 5), they can be run on the complete unannotated recordings for that site. The resulting predictions can be analyzed directly to study species occurrence patterns, or a subset of the predictions can be selected and manually verified to serve as training data for a subsequent active learning iteration.

`inference.py` builds sliding windows over raw audio files, computes mel spectrograms on the fly, runs the loaded checkpoint, and exports per-window predictions as a CSV. It supports binary and multiclass modes.

**Audio source options:**
- **Audio folder**: scans for all audio files, builds windows automatically, saves a `<dataset>_windows.json`
- **Spectrograms folder**: loads pre-computed `.npy` spectrograms directly, skipping the windowing and spectrogram computation steps
- **JSON file**: loads pre-built windows from a previous run
- **CSV file**: loads windows from a CSV with spectrogram paths

```bash
python inference.py --config data/data_config.yaml \
    --spectrograms_dir data/tuxedni_spectrograms \
    --checkpoint_binary checkpoints/tuxedni_binary-finetune/best.ckpt \
    --checkpoint_3class checkpoints/tuxedni_3class-finetune/best.ckpt \
    --output_csv inference/tuxedni_results.csv \
    --target_size 224 180 \
    --dataset tuxedni \
    --temperature 3 \
    --normalize
```

```bash
python inference.py --config data/data_config.yaml \
    --spectrograms_dir data/johnson_spectrograms \
    --checkpoint_binary checkpoints/johnson_binary-finetune/best.ckpt \
    --checkpoint_3class checkpoints/johnson_3class-finetune/best.ckpt \
    --output_csv inference/johnson_results.csv \
    --target_size 224 180 \
    --dataset johnson \
    --temperature 3 \
    --normalize
```
