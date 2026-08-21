# Improving the Models with Your Own Verified Data (Active Learning Guide)

**Audience:** biologists / analysts who are *not* machine-learning experts.
**Goal:** take a batch of detections you have listened to and corrected by hand, and use
them to make the models better at *your* recording site (e.g. Cook Inlet).

This process is called **fine-tuning** (a form of *active learning*): instead of training
a model from scratch, we take the existing trained models and nudge them with a small
amount of new, site-specific examples that you have verified.

---

## The big picture (read this first)

The models work in two stages, run one after the other ("cascade"):

1. **Binary model** — for each 2-second window of audio, decides *whale* vs *no whale*.
2. **3-class model** — for windows that contain a whale, decides the *species*
   (Humpback, Orca, or **Beluga**).

Depending on what's going wrong at your site, you'll fine-tune **one or both** of these
models — the helper script in Step 1 works out which from your corrections.

The full cycle looks like this, and you repeat it as you gather more verified data:

```
Run inference on new audio  ──▶  Listen & correct the results (your CSV)
        ▲                                     │
        │                                     ▼
Re-run inference with the   ◀──  Fine-tune the models on your corrections
improved models
```

---

## Before you start, you need three things

1. **Your verified CSV** — the inference output CSV *after* you have gone through it and
   corrected the labels by hand. Concretely: **keep** the model's original prediction
   columns (`pred_label`, or `pred_label_binary` / `pred_label_3class`) and **add one new
   column, `verified_label`**, holding the *correct* class for each window as a number:

   | `verified_label` | meaning |
   |------------------|---------|
   | `0` | No Whale |
   | `1` | Humpback |
   | `2` | Orca |
   | `3` | **Beluga** |

   This is the same `0/1/2/3` scheme the inference cascade uses for `pred_label`. Keeping the
   original predictions is what lets the helper script tell *false alarms* from *misses* and
   suggest the right fix.
2. **The spectrogram files** (`.npy`) for those windows. These were already created for you
   when you ran inference on the raw audio — look in
   `inference/<your_dataset>/spectrograms/`.
3. **The starting (base) checkpoints**, already in place:
   - `checkpoints/binary/best.ckpt`
   - `checkpoints/3class/best.ckpt`

---

## ⚠️ The most important rule: keep some data aside for testing

**Do not use all of your verified events for fine-tuning.** If you do, you will have no
honest way to know whether the model actually improved — you would only be able to check
it on the same examples it just learned from, which always looks good and tells you nothing.

Split your verified events into **three** groups:

| Group | Rough share | What it is used for |
|-------|-------------|---------------------|
| **Train** | ~70% | The examples the model actually learns from. |
| **Validation** ("val") | ~15% | Checked automatically *during* fine-tuning to pick the best version and stop at the right time. You do not look at this yourself. |
| **Test** | ~15% | **Locked away.** Never used in training. Only used at the very end to measure the real improvement. |

**Good news: you don't split the data by hand.** The helper script in Step 1
(`build_finetune_sets.py`) does the split for you, and handles the tricky parts:
- It splits into roughly **70 / 15 / 15** train / val / test.
- It keeps every window from the **same recording together** in one split (it groups by the
  `audio` column), so the model can't "cheat" by seeing the same call in both train and test.
- It uses a fixed random seed, so the split is reproducible from one round to the next.

You still need to *understand* why the test set matters — mainly so you don't delete it or
quietly reuse it for training — but the mechanics are handled for you.

---

## Step 1 — Turn your verified CSV into training files (`build_finetune_sets.py`)

You do **not** build the split files by hand. The helper script
[`build_finetune_sets.py`](../build_finetune_sets.py) does it for you: it reads your one
verified CSV (the one with the `verified_label` column you added) and writes the
train/val/test files in the exact format the trainer expects — columns `spec_name` and
`label`.

### 1a. Diagnose first — where is the model failing?

Before building anything, ask the script what's wrong with this batch. It compares the
model's predictions against your `verified_label`, prints detection precision/recall and
per-species confusion, and suggests a strategy:

```bash
python build_finetune_sets.py --verified_csv verified.csv --diagnose_only
```

### 1b. Pick a strategy

Different problems need different training sets, so instead of one fixed recipe the script
takes a **strategy** preset — a small YAML in
[`configs/active_learning/`](../configs/active_learning/). The diagnostic above suggests
one, but the choice is yours:

| Strategy | Use when… | Fine-tunes | Files it writes |
|----------|-----------|------------|-----------------|
| `hard_negatives` | Too many **false alarms** — noise called "whale" (low precision). | binary only | 3 × `*_binary.csv` |
| `add_positives` | **Missing real whales** — whales called "no whale" (low recall). | binary only | 3 × `*_binary.csv` |
| `species_correction` | Detection is fine, but **species are confused** (e.g. Beluga called Orca). | 3-class only | 3 × `*_3class.csv` |
| `balanced_refresh` | **General drift** at a new site, no single dominant problem. | both models | all six |

### 1c. Build the split files

```bash
python build_finetune_sets.py \
    --verified_csv verified.csv \
    --strategy hard_negatives \
    --output_dir data/cookinlet_splits
```

Depending on the strategy, this writes **three or six** files into
`data/cookinlet_splits/`:

```
data/cookinlet_splits/
├── train_binary.csv   val_binary.csv   test_binary.csv     (binary strategies + balanced_refresh)
└── train_3class.csv   val_3class.csv   test_3class.csv     (species_correction + balanced_refresh)
```

For reference, this is the `label` scheme the script writes into those files. **You don't
write these yourself** — the script derives them from your `verified_label`:

**Binary files** — contain *all* windows:
| label | meaning |
|-------|---------|
| `0` | no whale |
| `1` | whale (any species) |

**3-class files** — contain *only* whale windows, labelled by species:
| label | meaning |
|-------|---------|
| `0` | Humpback |
| `1` | Orca |
| `2` | **Beluga** |

> Mind the two schemes: your **input** `verified_label` uses `0/1/2/3` (0=No Whale … 3=Beluga),
> and the script converts that into the binary `0/1` and 3-class `0/1/2` files above.
> (In the 3-class files, Beluga is `2`, not `3`.)

---

## Step 2 — Fine-tune the model(s)

Run these one at a time. Each one starts from the base checkpoint and saves an improved,
site-adapted checkpoint. On a machine without a GPU this will be slow but still works.

**Only run the command for the model your strategy actually produced files for:**
`hard_negatives` and `add_positives` make **binary** files only, `species_correction` makes
**3-class** files only, and `balanced_refresh` makes both — so with `balanced_refresh` you
run both commands below.

```bash
# Fine-tune the binary (whale / no-whale) model
python train.py --config configs/config_binary.yaml \
    --train_csv data/cookinlet_splits/train_binary.csv \
    --val_csv   data/cookinlet_splits/val_binary.csv \
    --ckpt_path checkpoints/binary/best.ckpt \
    --finetune
```

```bash
# Fine-tune the 3-class (species) model
python train.py --config configs/config_3class.yaml \
    --train_csv data/cookinlet_splits/train_3class.csv \
    --val_csv   data/cookinlet_splits/val_3class.csv \
    --ckpt_path checkpoints/3class/best.ckpt \
    --finetune
```

Each run prints where it saved the new checkpoint (something like
`checkpoints/<experiment_name>/best.ckpt`). **Write these two paths down** — you need them
for the next steps.

---

## Step 3 — Measure the improvement on your held-out test set

This is where the test group you set aside earns its keep. Run each fine-tuned model in
"predict only" mode on the test CSVs, then compare.

```bash
# Binary model on the test set
python train.py --config configs/config_binary.yaml \
    --ckpt_path checkpoints/<your_binary_finetune>/best.ckpt \
    --test_csv data/cookinlet_splits/test_binary.csv \
    --exp_name cookinlet_test_finetuned \
    --output_csv binary.csv \
    --predict_only
```

```bash
# 3-class model on the test set
python train.py --config configs/config_3class.yaml \
    --ckpt_path checkpoints/<your_3class_finetune>/best.ckpt \
    --test_csv data/cookinlet_splits/test_3class.csv \
    --exp_name cookinlet_test_finetuned \
    --output_csv 3class.csv \
    --predict_only
```

```bash
# Combine the two stages and print the overall performance
python compare_models.py --binary_3class_only \
    --pred_binary test_results/cookinlet_test_finetuned/binary.csv \
    --pred_3class test_results/cookinlet_test_finetuned/3class.csv
```

If you want to prove the fine-tuning helped, run the **same** test with the original base
checkpoints too, and compare the two numbers.

---

## Step 4 — Use the improved models (and repeat)

Go back to the normal inference workflow, but point it at your **new** fine-tuned
checkpoints instead of the base ones:

```bash
python inference.py --config data/data_config.yaml \
    --spectrograms_dir inference/cookinlet/spectrograms \
    --checkpoint_binary checkpoints/<your_binary_finetune>/best.ckpt \
    --checkpoint_3class checkpoints/<your_3class_finetune>/best.ckpt \
    --output_csv inference/cookinlet/cascade_results.csv \
    --target_size 224 180 --dataset cookinlet --temperature 3 --normalize
```

In the output CSV, a **beluga detection is `pred_label == 3`**.

Review this new batch, correct it, add the corrections to your training files (keeping the
test set separate!), and fine-tune again. Each loop should make the model a little better
at your site.

---

## Common mistakes to avoid

- **Using every event for training.** Always keep a test set aside (see the rule above).
- **Mixing the same call into train and test.** Keep windows from one recording/event in
  one group.
- **Forgetting to switch checkpoints.** After fine-tuning, inference must point at the
  *new* checkpoint paths, not the base ones.
- **Too few examples.** Fine-tuning with only a handful of events won't help much. More
  verified data — especially of the cases the model currently gets wrong — is what moves
  the needle.

---

## Getting help

- The overall pipeline (preparing data, training, inference) is documented in the main
  [`README.md`](../README.md).
- The trained checkpoints and the annotation labels are published on Zenodo:
  <https://zenodo.org/records/19490105>.
- The "turn my verified CSV into training files" step is done by
  [`build_finetune_sets.py`](../build_finetune_sets.py) (Step 1). Run it with
  `--diagnose_only` first to see what it recommends. The strategy presets it uses live in
  [`configs/active_learning/`](../configs/active_learning/) and are plain YAML you can copy
  and tweak for your own situation.
