# Bioacoustics Review

A generic Streamlit app for reviewing per-segment classifier predictions on
spectrograms: page through the predictions, listen to the audio, and assign
a `manual_verif` ground-truth label. Built originally for the Cook Inlet
Belugas project but driven by `frontend/config.py` so it can be adapted to
any spectrogram-based bioacoustic model.

## Launch

```bash
pip install -r requirements.txt          # at repo root
export AUDIO_ROOT=/path/to/wav/files     # required for audio + 10-s view
export INFERENCE_DIR=/path/to/csvs       # optional; powers the CSV picker
export DEFAULT_CSV=/path/to/run.csv      # optional; pre-selected on launch
streamlit run frontend/app.py --server.port 8501
```

All paths come from environment variables — no path is hardcoded, so the
same checkout works for any user on any machine.

## Configuring for your project

Edit `frontend/config.py`. The constants there cover everything project-
specific:

| Section | What to change |
|---|---|
| Paths | Defaults if you'd rather not use env vars |
| Audio / spectrogram | `SAMPLE_RATE`, `HIGHPASS_CUTOFF_HZ`, mel parameters |
| CSV schema | Column names (`audio`, `start(s)`, `end(s)`, `pred_label`, `manual_verif`, …) |
| Class taxonomy | `PRED_LABELS` (int → name), title colors, probability bar colors |
| Manual-verification labels | `MANUAL_VERIF_LABELS` — the buttons users click + their keyboard shortcuts |

## Workflow

- The source CSV is read-only. Reviewed labels are written to
  `frontend/reviews/<csv_stem>_<user>_reviewed.csv` — concurrent reviewers
  on different clones cannot stomp each other.
- Every `BACKUP_EVERY_N_SAVES` saves (default 5), the reviewed CSV is
  copied to `frontend/reviews/backups/<stem>_<timestamp>.csv`.
- If `EAR.LOG` is present (or `EAR_LOG_PATH` points to one), the recording
  start datetime is shown under the spectrogram. Optional.

## Keyboard shortcuts (Review page)

- `← / →` — Prev / Next
- `1` — auto-contrast · `3` — noise reduction · `p` — high-pass filter
- `2` — toggle expanded (10-s) view
- Per-label keys are configured in `MANUAL_VERIF_LABELS` (Cook Inlet
  defaults: `b` Beluga, `h` Humpback, `o` Orca, `n` Noise, `z` off_effort,
  `u` Unsure)
