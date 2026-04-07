# Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes

This repository contains the code and experiment pipelines to reproduce the results in the publication "Adaptive acoustic monitoring for endangered Cook Inlet beluga whales in complex soundscapes" by Castellote et al. (2026). This work is framed within the long-term Cook Inelt Beluga Acoustics (CIBA) Program led by the Marine Mammal Laboratory, Alaska Fisheries Science Center, and the Division of Wildlife Conservation, Alaska Department of Fish and Game. It implements an end-to-end workflow for the analysis of archival sound recordings from moored instruemnts in Cook Inlet, Alaska, including spectrogram generation, a two-stage deep learning architecture for cetacean signal detection and species classification (beluga, humpback whale, killer whale), and an active-learning loop for domain adaptation to new soundscapes as mooring deployments expand to unsampled areas. The repository is organized to support training, evaluation, inference on long-duration recordings, and replication of the final framework proposed in the manuscript.

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
├── annotations_combined.json
├── windows_mapping_0.4overlap.json
├── mel_spectrograms_multiclass/   # .npy spectrograms
├── splits_4class/                 # 4-class CSVs (noise + 3 species)
│   ├── train_split.csv
│   ├── val_split.csv
│   └── test_split.csv
├── splits_3class/                 # 3-class CSVs (species only)
│   └── ...
└── splits_binary/                 # Binary CSVs (whale vs. no-whale)
    └── ...
```

### 3. Train Base Models

All model and training parameters are defined in YAML configs under `configs/`. Only runtime arguments (split paths, checkpoint path) are passed via CLI. Config files can be referenced by filename alone — the script resolves them from the `configs/` directory automatically.

```bash
# Binary classification — Stage 1 (whale / no-whale), ResNet18
python train.py --config config_binary.yaml \
    --train_csv data/splits_exp1/splits_binary/train_split.csv \
    --val_csv data/splits_exp1/splits_binary/val_split.csv \
    --test_csv data/splits_exp1/splits_binary/test_split.csv
```

```bash
# 3-class species classification — Stage 2 (humpback / orca / beluga), ResNet34
python train.py --config config_3class.yaml \
    --train_csv data/splits_exp1/splits_3class/train_split.csv \
    --val_csv data/splits_exp1/splits_3class/val_split.csv \
    --test_csv data/splits_exp1/splits_3class/test_split.csv
```

```bash
# 4-class - Stage 1 (no-whale + 3 species), ResNet34
python train.py --config config_4class_75.yaml \
    --train_csv data/splits_exp1/splits_4class/train_split.csv \
    --val_csv data/splits_exp1/splits_4class/val_split.csv \
    --test_csv data/splits_exp1/splits_4class/test_split.csv
```

```bash
# 4-class — Stage 2 (no-whale + 3 species), ResNet34
python train.py --config config_4class_25.yaml \
    --train_csv data/splits_exp1/splits_4class_25/train_split.csv \
    --val_csv data/splits_exp1/splits_4class_25/val_split.csv \
    --test_csv data/splits_exp1/splits_4class_25/test_split.csv
```

**Evaluate from a saved checkpoint (no training):**

#### Base Models - Tuxedni Channel

```bash
python train.py --config config_binary_exp1.yaml --test_csv /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/data/tuxedni_splits/final_test.csv --ckpt_path /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/checkpoints/binary_exp1/best.ckpt --spectrograms_dir ./data/tuxedni_final_test_spectrograms --exp_name tuxedni_final_test --output_csv binary.csv --predict_only
```

```bash
python train.py --config config_3class_exp1.yaml --test_csv /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/data/tuxedni_splits/final_test.csv --ckpt_path /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/checkpoints/3class_exp1/best.ckpt --spectrograms_dir ./data/tuxedni_final_test_spectrograms --exp_name tuxedni_final_test --output_csv 3class.csv --predict_only
```

```bash
python compare_models.py --binary_3class_only --pred_binary /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/test_results/tuxedni_final_test/binary.csv --pred_3class /home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/test_results/tuxedni_final_test/3class.csv
```

#### Base Models - Johnson River

```bash
python train.py --config config_binary_exp1.yaml --test_csv ./data/johnson_splits/final_test.csv --ckpt_path ./checkpoints/binary_exp1/best.ckpt --spectrograms_dir ./data/johnson_final_test_spectrograms --exp_name johnson_final_test --output_csv binary.csv --predict_only
```

```bash
python train.py --config config_3class_exp1.yaml --test_csv ./data/johnson_splits/final_test.csv --ckpt_path ./checkpoints/3class_exp1/best.ckpt --spectrograms_dir ./data/johnson_final_test_spectrograms --exp_name johnson_final_test --output_csv 3class.csv --predict_only
```

```bash
python compare_models.py --binary_3class_only --pred_binary ./test_results/johnson_final_test/binary.csv --pred_3class ./test_results/johnson_final_test/3class.csv
```

The `--predict_only` flag skips metric computation and only exports a predictions CSV, which is useful when the test label space differs from the model's label space.

### 4. Active Learning Training

#### Base Modes - Tuxedni Channel

```bash
python train.py --config configs/tuxedni/binary.yaml \
    --train_csv data/tuxedni_splits/train_binary.csv \
    --val_csv data/splits_exp1/splits_binary/val_split_binary.csv \
    --val_spectrograms_dir ./data/mel_spectrograms_multiclass  \
    --ckpt_path ./checkpoints/binary_exp1/best.ckpt \
    --finetune
```

```bash
python train.py --config config_binary_exp1.yaml --test_csv data/tuxedni_splits/final_test.csv --ckpt_path checkpoints/tuxedni_binary-finetune/best.ckpt --spectrograms_dir ./data/tuxedni_final_test_spectrograms --exp_name tuxedni_final_test_finetuned --output_csv binary.csv --predict_only
```

```bash
python train.py --config configs/tuxedni/3class.yaml \
    --train_csv data/tuxedni_splits/train_3class.csv \
    --val_csv data/splits_exp1/splits_3class/val_split.csv \
    --val_spectrograms_dir ./data/mel_spectrograms_multiclass  \
    --ckpt_path ./checkpoints/3class_exp1/best.ckpt \
    --finetune
```

```bash
python train.py --config config_3class_exp1.yaml --test_csv ./data/tuxedni_splits/final_test.csv --ckpt_path ./checkpoints/tuxedni_3class-finetune/best.ckpt --spectrograms_dir ./data/tuxedni_final_test_spectrograms --exp_name tuxedni_final_test_finetuned --output_csv 3class.csv --predict_only
```

```bash
python compare_models.py --binary_3class_only --pred_binary ./test_results/tuxedni_final_test_finetuned/binary.csv --pred_3class ./test_results/tuxedni_final_test_finetuned/3class.csv
```

#### Base Modes - Johnson River

```bash
python train.py --config configs/johnson/binary.yaml \
    --train_csv data/johnson_splits/train_binary.csv \
    --val_csv data/splits_exp1/splits_binary/val_split_binary.csv \
    --val_spectrograms_dir ./data/mel_spectrograms_multiclass  \
    --ckpt_path ./checkpoints/binary_exp1/best.ckpt \
    --finetune
```

```bash
python train.py --config config_binary_exp1.yaml --test_csv data/johnson_splits/final_test.csv --ckpt_path checkpoints/johnson_binary-finetune/best.ckpt --spectrograms_dir ./data/johnson_final_test_spectrograms --exp_name johnson_final_test_finetuned --output_csv binary.csv --predict_only
```

```bash
python train.py --config configs/johnson/3class.yaml \
    --train_csv data/johnson_splits/train_3class.csv \
    --val_csv data/splits_exp1/splits_3class/val_split.csv \
    --val_spectrograms_dir ./data/mel_spectrograms_multiclass  \
    --ckpt_path ./checkpoints/3class_exp1/best.ckpt \
    --finetune
```

```bash
python train.py --config config_3class_exp1.yaml --test_csv ./data/johnson_splits/final_test.csv --ckpt_path ./checkpoints/johnson_3class-finetune/best.ckpt --spectrograms_dir ./data/johnson_final_test_spectrograms --exp_name johnson_final_test_finetuned --output_csv 3class.csv --predict_only
```

```bash
python compare_models.py --binary_3class_only --pred_binary ./test_results/johnson_final_test_finetuned/binary.csv --pred_3class ./test_results/johnson_final_test_finetuned/3class.csv
```

### 5. Inference on Long-Duration Recordings

`inference.py` builds sliding windows over raw audio files, computes mel spectrograms on the fly, runs the loaded checkpoint, and exports per-window predictions as a CSV. It supports binary and multiclass modes.

**Audio source options:**
- **Folder**: scans for all audio files, builds windows automatically, saves a `<dataset>_windows.json`
- **JSON file**: loads pre-built windows from a previous run
- **CSV file**: loads windows from a CSV with spectrogram paths

```bash
python inference.py --spectrograms_dir /home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/tuxedni_spectrograms --checkpoint_binary checkpoints/tuxedni_binary-finetune/best.ckpt --checkpoint_3class checkpoints/tuxedni_3class-finetune/best-v1.ckpt --sample_rate 24000 --target_size 224 180 --dataset tuxedni --output_csv ./inference/tuxedni_binary+3class_complete_results_temp3.csv --temperature 3 
```

```bash
python inference.py --spectrograms_dir /home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/Johnson_River_Cook_Inelt_09-2022_06-2023/johnson_spectrograms --checkpoint_binary checkpoints/johnson_binary-finetune/best.ckpt --checkpoint_3class checkpoints/johnson_3class-finetune/last-v3.ckpt --sample_rate 24000 --target_size 224 180 --dataset tuxedni --output_csv ./inference/johnson_binary+3class_complete_results_temp3.csv --temperature 3 
```
