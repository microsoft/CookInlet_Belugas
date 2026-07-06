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

You will fine-tune **both** models, because both benefit from local examples.

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
   corrected the labels by hand (kept the correct ones, fixed the wrong ones).
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

Practical tips for splitting well:
- Split **randomly**, but keep a mix of species and of whale/no-whale in each group.
- If several windows come from the **same continuous recording / same call**, try to keep
  them together in the *same* group. Otherwise the model can "cheat" by memorising a sound
  that appears in both train and test.
- Keep the test group **untouched** across rounds if you can, so improvements are comparable
  over time.

---

## Step 1 — Put your CSV into the format the trainer expects

The training script reads two columns from each CSV:

- **`spec_name`** — the path to that window's `.npy` spectrogram file
  (you already have these from inference).
- **`label`** — the *verified* class, written as a number:

  **For the binary CSVs:**
  | label | meaning |
  |-------|---------|
  | `0` | no whale |
  | `1` | whale (any species) |

  **For the 3-class CSVs** (only include windows that *do* contain a whale):
  | label | meaning |
  |-------|---------|
  | `0` | Humpback |
  | `1` | Orca |
  | `2` | **Beluga** |

So from your one verified CSV you will produce **six** files:

```
data/cookinlet_splits/
├── train_binary.csv     val_binary.csv     test_binary.csv
└── train_3class.csv     val_3class.csv     test_3class.csv
```

> The binary files contain *all* windows (whale and no-whale).
> The 3-class files contain *only* the whale windows, labelled by species.

*(A helper script can generate these six files from your verified CSV automatically —
ask the developer, or see "Getting help" below.)*

---

## Step 2 — Fine-tune the two models

Run these one at a time. Each one starts from the base checkpoint and saves an improved,
site-adapted checkpoint. On a machine without a GPU this will be slow but still works.

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
- If the "format your CSV into six split files" step is unclear, ask the developer for the
  conversion helper script — that step is the only part that isn't a single command.
