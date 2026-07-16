# Cook Inlet Whale-Call Inference — Quick Manual

A plain-language guide to run the model on your own `.wav` audio and get, for
every short slice, a prediction of **which whale** (if any) is calling:
**No Whale**, **Humpback**, **Orca**, or **Beluga**.

You do **not** need to understand the training code to use this. To just score
audio, jump to [Section 3](#3-run-inference). If you want to improve the models
on your own site, see the separate [Active Learning Guide](docs/active_learning_guide.md).

> This is the updated pipeline (Castellote et al. 2026). It replaces the older
> single-species Beluga ensemble (Ming et al. 2019). The old model answered one
> question — *"Beluga: yes/no?"*. This one runs a **two-stage cascade** that
> first asks *"is there a whale?"* and then *"which species?"*.

---

## 1. What this thing does (in one picture)

```
  your .wav file
        |
        v
  cut into 2-second slices (0.4 s overlap)
        |
        v
  each slice -> a mel SPECTROGRAM (.npy array, computed on GPU)
        |
        v
  STAGE 1  binary model (ResNet18):   whale  vs  no-whale
        |
        |  (only slices that pass stage 1 continue)
        v
  STAGE 2  3-class model (ResNet34):  Humpback | Orca | Beluga
        |
        v
  one label per slice:  0=No Whale  1=Humpback  2=Orca  3=Beluga
```

- Stage 1 is a **whale / no-whale detector**. Stage 2 names the **species**.
- If stage 1 says "no whale", the final label is `0` (No Whale) regardless of stage 2.
- **A Beluga detection is `pred_label == 3`.**
- Both models are **already trained** — nothing to train to run inference. The
  trained checkpoints are **not** in this repo; you download them once from
  Zenodo (see [Section 2c](#2c-get-the-trained-model-checkpoints)).

Why two stages instead of one 4-class model? The soundscape is dominated by
non-whale noise, so a dedicated detector (stage 1) is more robust, and a
smaller species model (stage 2) only has to separate the three whales. See the
paper and `compare_models.py` for the comparison.

---

## 2. One-time setup

You need three things: the **code** (this repo), a **Python environment** with the
dependencies, and the **trained checkpoints** (downloaded separately). Nothing is
pre-built — the steps below create everything from a clean clone.

### 2a. Create the Python environment

**Use Python 3.11 or 3.12** (the pinned dependencies have no wheels for Python 3.13+).

```powershell
# from the repo root, after: git clone ... && cd CookInlet_Belugas
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
```

That's the whole install. On **Linux** this already includes GPU (CUDA) PyTorch;
on **Windows/macOS** it installs the CPU build — add the GPU build in Section 2b.

Activate the environment (`.\venv\Scripts\Activate.ps1`) in every new shell before
running the scripts, or call the venv's Python directly:
`& .\venv\Scripts\python.exe inference.py --help`.

### 2b. GPU acceleration (NVIDIA) — optional but recommended

Native-Windows PyTorch supports CUDA (no WSL2 needed). To use your GPU on
Windows/macOS, install the CUDA build of the torch packages from the PyTorch
index — match the CUDA tag to your driver (`cu126` shown here):

```powershell
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 `
    --index-url https://download.pytorch.org/whl/cu126
```

Verify CUDA is live:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`2.7.1+cu126 True` = GPU ready. `2.7.1 False` (or `+cpu`) = CPU build.

- The GPU accelerates **both** the mel-spectrogram computation
  (`compute_mel_spectrograms_gpu`) **and** the model forward pass, so a GPU
  meaningfully speeds up the whole run.
- No GPU? Everything still runs on CPU — the scripts print
  `CUDA not available, switching to CPU` and keep going. Just pass `--device cpu`.

### 2c. Get the trained model checkpoints

The two cascade checkpoints are **not** stored in this repo (`checkpoints/` is
git-ignored). Download them from Zenodo and place them where the commands expect:

> **Zenodo:** <https://zenodo.org/records/19490105> (trained checkpoints + annotation labels)

```
checkpoints/
    binary/best.ckpt     <- Stage 1: whale / no-whale (ResNet18)
    3class/best.ckpt     <- Stage 2: Humpback / Orca / Beluga (ResNet34)
```

These are the base (Cook Inlet) models. If you fine-tune to a new site you'll get
**new** checkpoints under `checkpoints/<experiment>-finetune/` — see the
[Active Learning Guide](docs/active_learning_guide.md). In every command below,
point `--checkpoint_binary` / `--checkpoint_3class` at whichever checkpoints you
want to use (base or fine-tuned).

---

## 3. Run inference

`inference.py` does the whole pipeline for you. For the published cascade (the
paper's setup, and what you almost always want) you have two options: the
**one-command wrapper** (raw audio → cascade), or **cascade mode** directly if
you already have spectrograms.

### The easy path: one command from raw audio (`run_cascade.py`)

`run_cascade.py` is a thin wrapper around `inference.py`: it builds the mel
spectrograms from your `.wav` files, then runs the two-stage cascade on them —
no separate spectrogram step to remember.

```powershell
python run_cascade.py `
    --config data/data_config.yaml `
    --audios_source "C:\path\to\folder_of_wavs" `
    --checkpoint_binary checkpoints/binary/best.ckpt `
    --checkpoint_3class checkpoints/3class/best.ckpt `
    --dataset mysite
```

That's the whole thing. It writes:
- the spectrograms to `inference/mysite/spectrograms/` (reused on re-runs; pass
  `--force` to recompute), and
- the results to `inference/mysite/cascade_results.csv` (override with `--output_csv`).

It defaults to the correct settings for the base checkpoints
(`--target_size 224 180`, `--temperature 3`, normalize on). Override any of
them with the same flags as below (`--conf_threshold`, `--batch_size`,
`--device`, `--no_normalize`, …). Point `--checkpoint_binary` /
`--checkpoint_3class` at your fine-tuned checkpoints once you've adapted to a site.

### The explicit path: cascade mode (both species stages)

Cascade mode runs stage 1 **and** stage 2 and writes one combined CSV. It needs
a folder of **pre-computed `.npy` spectrograms** plus the two checkpoints:

```powershell
python inference.py --config data/data_config.yaml `
    --spectrograms_dir data/tuxedni_spectrograms `
    --checkpoint_binary checkpoints/binary/best.ckpt `
    --checkpoint_3class checkpoints/3class/best.ckpt `
    --output_csv inference/tuxedni_results.csv `
    --target_size 224 180 --dataset tuxedni --temperature 3 --normalize
```

(Point `--checkpoint_binary` / `--checkpoint_3class` at your fine-tuned
checkpoints instead of the base ones once you've adapted to a site.)

**"But I only have raw `.wav` files, not spectrograms."** Use `run_cascade.py`
above — it builds them for you. (Or build them yourself with the single-model
path below, or `prepare_dataset.py --steps spectrograms`, then point
`--spectrograms_dir` at that folder.)

### The simple path: single model on raw audio

Point `--audios_source` at a folder of `.wav` files (or a single file). This
runs **one** model, and — as a side effect — computes and **saves the
spectrograms** you can then feed to cascade mode:

```powershell
# Stage-1 detector only, straight from raw audio
python inference.py --config data/data_config.yaml `
    --checkpoint checkpoints/binary/best.ckpt `
    --audios_source "C:\path\to\folder_of_wavs" `
    --dataset tuxedni --temperature 3 --normalize
```

`--audios_source` also accepts a `.json` of pre-built windows or a `.csv`
manifest with spectrogram paths, if you have them from a previous run.

### The flags that matter

| Flag | Meaning | Typical |
|------|---------|---------|
| `--config` | YAML with audio + spectrogram params (**required**). Use `data/data_config.yaml`. | `data/data_config.yaml` |
| `--spectrograms_dir` | Folder of pre-computed `.npy` spectrograms → **turns on cascade mode**. | your specs |
| `--checkpoint_binary` / `--checkpoint_3class` | The two cascade checkpoints. | base or fine-tuned |
| `--audios_source` | Folder of `.wav` (or `.json`/`.csv`) → single-model mode. | your audio |
| `--checkpoint` | Single model checkpoint (single-model mode). | one `.ckpt` |
| `--output_csv` | Where to write the cascade CSV. | `inference/<site>_results.csv` |
| `--target_size H W` | Spectrogram size the models expect. **Use `224 180`** for these checkpoints. | `224 180` |
| `--temperature T` | Softens probabilities (calibration). The paper uses **3**. | `3` |
| `--normalize` | Normalize each spectrogram before the model. Match how the model was trained — **keep it on** for these checkpoints. | on |
| `--conf_threshold X` | Stage-1 probability cut-off for "whale". Higher = fewer, stricter detections. | `0.5` |
| `--batch_size N` | Slices per model call (GPU/CPU utilisation). | `64` |
| `--device` | `cuda` or `cpu`. Auto-falls back to CPU if no GPU. | `cuda` |
| `--dataset` | Name used for the output subfolder / windows file. | `tuxedni` |
| `--num_classes` / `--class_names` | Override the class setup (single-model mode); otherwise read from `--config`. | from config |

> Audio and spectrogram parameters (sample rate, window size, `n_fft`, `n_mels`,
> …) come from `--config` so they stay **identical** to what the models were
> trained on. Don't change them unless you know what you're doing — see the
> gotchas.

### What you get (cascade mode)

One CSV (`--output_csv`), one row per slice:

```
file_path, audio, start(s), end(s), pred_label, pred_label_binary, prob_class_0, confidence_binary, pred_label_3class, prob_class_1, prob_class_2, prob_class_3
```

- `audio` = source recording name; `start(s)` / `end(s)` = slice bounds in seconds.
- **`pred_label`** = the final answer: **`0`=No Whale, `1`=Humpback, `2`=Orca, `3`=Beluga**.
- `pred_label_binary` = stage-1 result (`0`=no whale, `1`=whale).
- `prob_class_0` = probability of **no whale** (`1 − stage-1 whale probability`).
- `confidence_binary` = how far stage 1 is from the 0.5 fence (`0`=unsure, `1`=certain).
- `pred_label_3class` = stage-2 species (remapped to `1`=Humpback, `2`=Orca, `3`=Beluga).
- `prob_class_1` / `prob_class_2` / `prob_class_3` = stage-2 species probabilities
  (Humpback / Orca / Beluga; they sum to ~1).

> **To find Belugas: filter `pred_label == 3`.** For a ranked list, sort those
> rows by `prob_class_3` (and/or `confidence_binary`) descending.

Single-model mode instead writes `inference/<dataset>/binary_inference_results.csv`
(or `multiclass_inference_results.csv`), with `prediction`, `probability`, and
`confidence` columns.

---

## 4. Things to know / gotchas

- **Input is `.wav`.** The window builder reads audio with soundfile/librosa and
  resamples to the config `sample_rate` (24 kHz). Convert mp3/flac to wav first.
- **Keep `--target_size 224 180`, `--normalize`, and `--temperature 3`** for the
  base checkpoints. These match how the models were trained/calibrated;
  changing them silently degrades results.
- **Don't change the spectrogram recipe.** `n_fft`, `hop_length`, `n_mels`,
  `top_db`, etc. live in `data/data_config.yaml` and must match training. The
  models were trained on exactly those mel spectrograms.
- **Cascade mode needs spectrograms on disk.** It does *not* read raw `.wav`
  directly — build the `.npy` spectrograms first (single-model raw-audio run, or
  `prepare_dataset.py --steps spectrograms`), then run the cascade on that folder.
- **Spectrogram filename convention** is `{audio}_{start_samples}_{end_samples}.npy`.
  `inference.py` parses the start/end sample offsets back out of the name, so
  don't rename the files. (A legacy `sid…_idx…` naming is also recognised.)
- **`No Whale` is a rejection, not a species claim.** `pred_label == 0` just
  means "no confident whale call in this slice".
- **New site behaving oddly?** The base models were trained on Cook Inlet. On a
  very different soundscape you may see more false alarms or species confusion —
  that's exactly what the **fine-tuning / active-learning** loop is for. See the
  [Active Learning Guide](docs/active_learning_guide.md).

---

## 5. How this maps to the rest of the repo (for reference)

You don't need these just to score audio, but here's the full pipeline:

| Script | Purpose |
|--------|---------|
| `prepare_dataset.py` | Build the training dataset from annotations: `stats` → `windows` → `spectrograms` → `splits`. Driven by `data/data_config.yaml`. |
| `train.py` | Train / fine-tune / evaluate a model. All model params come from a `configs/*.yaml`; only split paths + checkpoint come from the CLI. |
| `inference.py` | Score audio (this manual). Cascade or single-model. |
| `run_cascade.py` | One-command wrapper: raw audio → spectrograms → cascade CSV. |
| `compare_models.py` | Combine stage-1 + stage-2 prediction CSVs and report precision/recall/F1 (used to evaluate a site). |
| `build_finetune_sets.py` | Turn a hand-verified CSV into train/val/test files for fine-tuning (active learning). |
| `configs/config_binary.yaml` | Stage-1 (whale/no-whale, ResNet18) config. |
| `configs/config_3class.yaml` | Stage-2 (species, ResNet34) config. |
| `configs/config_4class_75.yaml`, `configs/config_4class_25.yaml` | Single 4-class model configs (the alternative to the cascade; see `compare_models.py`). |
| `configs/tuxedni/`, `configs/johnson/`, `configs/active_learning/` | Fine-tuning presets for the paper's two sites and for common active-learning strategies. |

The full training → evaluation → fine-tuning → inference workflow, including the
exact Tuxedni Channel and Johnson River commands from the paper, is in the main
[`README.md`](README.md).

### The four label schemes (don't mix them up)
| Where | Scheme |
|-------|--------|
| Cascade `pred_label` (inference output) | `0`=No Whale, `1`=Humpback, `2`=Orca, `3`=Beluga |
| `verified_label` you add when correcting (active learning input) | same `0/1/2/3` |
| Binary training/`label` files | `0`=no whale, `1`=whale |
| 3-class training/`label` files | `0`=Humpback, `1`=Orca, `2`=Beluga *(Beluga is `2` here, not `3`)* |

---

*Cite: Castellote et al. 2026, "Adaptive acoustic monitoring for endangered Cook
Inlet beluga whales in complex soundscapes", Marine Mammal Science
(doi:10.1111/mms.70213). See `CITATION.cff`.*
