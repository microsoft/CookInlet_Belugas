# Cook Inlet Belugas — Frontend

Streamlit multi-page app for the Active Learning review workflow.

## Pages

| Page | URL path | Purpose |
|---|---|---|
| Home | `/` | Navigation explainer |
| Review | `/Review` | Step through spectrogram predictions and assign `manual_verif` labels |
| AL Targets | `/AL_Targets` | Check whether any threshold meets your P/R/F1 requirements |

## Launch

```bash
conda activate bioacustics
cd /home/v-manoloc/shared/v-manoloc/CookInlet_Belugas
streamlit run frontend/app.py --server.port 8501
```

Then open `http://localhost:8501` in your browser.

## Where review CSVs are saved

Reviewed labels are written to:
```
frontend/reviews/<input_basename>_<username>_reviewed.csv
```

This file is created on first save and is **separate from the source CSV** — the source is never modified.

If you close and reopen the app, the Review page will automatically load from your reviewed file (if it exists) so you pick up where you left off.

## Data paths (hard-coded defaults)

| Resource | Path |
|---|---|
| Demo prediction CSV | `/home/v-manoloc/shared/v-manoloc/tuxedni_results_1stAL_round_rev.csv` |
| Threshold sweep CSVs | `/home/v-manoloc/shared/v-druizlopez/CookInlet_Belugas/inference/` |
| Tuxedni WAV files | `/home/v-manoloc/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/all_audios/` |

## Notes

- The app remaps `.npy` paths that contain `/home/v-druizlopez/shared/` to the local shared mount automatically.
- Audio requires `soundfile` (already in `requirements.txt`). If a `.wav` can't be found the row still loads with an "Audio unavailable" notice.
- Large CSVs (>10 k rows) will trigger a warning but still load.
