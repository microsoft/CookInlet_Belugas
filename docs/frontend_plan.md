# Plan: Bioacoustics Pipeline Frontend (2-hour vibe-coding session)

## Context

The Cook Inlet Belugas project trains ResNet classifiers on whale-call spectrograms and runs an active-learning loop where a biologist reviews model predictions and assigns a `manual_verif` ground-truth label. Today that review happens in spreadsheets; the AL Round-2 scripts (`al_round2_*.py`) generate prediction CSVs and threshold sweeps, but no UI exists for the biologist to (a) review/relabel predictions or (b) decide whether a candidate model meets his performance bar.

The goal of this 2-hour session is to ship the two highest-leverage pieces of that workflow:

1. **Spectrogram review & relabel UI** — the actual bottleneck of the AL loop.
2. **AL targets tool** — answer "does any threshold/model meet my P/R/F1 requirements?" against the existing threshold-sweep CSVs.

A data-prep frontend (originally requested as component #1) is cut: Person A handles splits manually and a UI for it would not save time. A results-synthesis report (originally #4) is a stretch goal only — the AL Targets page already covers most of its analytical value.

**Decisions confirmed with user**:
- Scope: components #2 + #3 only.
- Audio playback: **included** (slice from source `.wav` via `soundfile`).
- Demo data: existing reviewed CSV `inference/tuxedni_results_1stAL_round_rev.csv` (1,095 rows, `manual_verif` already partially populated) — no pre-curation needed.
- Host: shared VM, both users have their own clone, shared data at `/home/v-druizlopez/shared/`.

## Recommended Approach

**Stack**: Streamlit multi-page app. Single dependency (`pip install streamlit`), Python-only, hot-reload, multi-page via `pages/`. Gradio is weaker for the stateful per-row review UX; Jupyter widgets put Person B inside a programming surface; static HTML is the wrong tool for 2 hours.

**Architecture**: All new code under `frontend/`. The app reads existing CSVs (read-only on the input) and writes review results to a separate file `frontend/reviews/<input_basename>_<user>_reviewed.csv` so concurrent reviewers (Person A and Person B on separate clones) cannot stomp each other.

## File Structure

| File | Purpose | LOC | Reuses |
|---|---|---|---|
| `frontend/app.py` | Streamlit entry point. Title + nav explainer. | ~30 | — |
| `frontend/pages/1_Review.py` | Component #2. Spectrogram + audio + radio buttons + Save. Filters in sidebar (only-unverified, by `pred_label`, by `prob_whale` range). Prev/Next nav. | ~180 | `data/plot_spectrograms.py:hz_to_mel`, `CATEGORY_MAP`, `LABEL_COLORS`, `SAMPLE_RATE` |
| `frontend/pages/2_AL_Targets.py` | Component #3. Multi-select threshold-sweep CSVs from `inference/`. Sliders: min beluga P/R/F1, min macro_F1. Per-CSV verdict + comparison table. | ~120 | Reads `inference/final_test_round{1,2}_threshold_sweep.csv`, `inference/phase1_e14xr13c_threshold_sweep.csv` directly |
| `frontend/spec_render.py` | Pure helper: `(npy_path) -> Figure` with mel-axis Hz ticks. Used by review page. | ~50 | Mel→Hz tick logic from `data/plot_spectrograms.py` |
| `frontend/audio_io.py` | `load_audio_slice(audio_basename, start_s, end_s) -> (ndarray, sr)`. Resolves `audio` column to source `.wav` under `/home/v-druizlopez/shared/...`. Returns `None` gracefully if not found. | ~40 | `soundfile` (already in requirements.txt) |
| `frontend/csv_io.py` | `load_predictions(path)`, `upsert_review(reviewed_path, row_key, manual_verif_value)`. Writes via temp+rename for atomicity. Adds `manual_verif` column if missing. | ~50 | — |
| `frontend/README.md` | Launch command, layout, where review CSVs are saved. | ~30 | — |
| `requirements.txt` | Append `streamlit` and `matplotlib` (matplotlib is currently transitive only). | +2 lines | — |

**Launch**: `conda activate bioacustics && streamlit run frontend/app.py --server.port 8501`

## Time Budget (15-min blocks)

| Block | Person A (codes) | Person B (no code) | Must / Stretch |
|---|---|---|---|
| 0:00–0:10 | `pip install streamlit matplotlib` in `bioacustics`. Smoke tests: `np.load(file_path)` on row 0 of demo CSV; `sf.read(...)` slice on the matching `.wav` (locate it under `/home/v-druizlopez/shared/...`). Create branch `frontend-spike`. | Hand A target numbers + paper UI sketch + his preferred radio-button labels (e.g., is it `off_effort` or `out of effort`?). | Must |
| 0:10–0:25 | App skeleton (`app.py`, `pages/`, `csv_io.py`). Sidebar CSV picker defaulting to `inference/tuxedni_results_1stAL_round_rev.csv`. Cached CSV load. | Watch the picker; flag if path layout is unintuitive. | Must |
| 0:25–0:45 | `spec_render.py`: render `.npy` to a Figure with Hz ticks. Wire into Review page above radio buttons. Show `prob_whale`/`prob_humpback`/`prob_orca`/`prob_beluga` and `pred_label` text panel. | Drive on row 0–5. Verify spectrogram is readable, ticks are correct, probability layout is clear. | Must |
| 0:45–1:00 | Radio buttons {beluga, noise, off_effort, humpback, orca, ""}. Save button → `csv_io.upsert_review`. Prev/Next nav. Row counter "47 of 1,095". | Review 10 rows; confirm save round-trip by reopening reviewed CSV. | Must |
| 1:00–1:15 | `audio_io.py` + `st.audio` widget on Review page. Cache the slice per-row. Graceful "audio unavailable" placeholder if `.wav` not found. | Validate audio sync with spectrogram window; flag any clipping/quality issues. | Must |
| 1:15–1:25 | Sidebar filters: only-unverified toggle (default ON), `pred_label` multiselect, `prob_whale` slider. | Drive a 20-row review with filters; report any filter Person B finds confusing. | Must |
| 1:25–1:45 | `pages/2_AL_Targets.py`: load all `*_threshold_sweep.csv` files, multi-select, sliders for min beluga P/R/F1, table of passing rows per file, verdict text ("best at threshold=0.42, beluga F1=0.74" or "no threshold meets target — closest is …"). | Provide his actual target numbers as test inputs. Confirm verdict text is actionable. | Must |
| 1:45–2:00 | Demo dry-run from a clean tab. One bug-fix pass. Commit to `frontend-spike`. | Drive the demo end-to-end without A's hands on keyboard. | Must |
| **+15 if available** | Stretch: `pages/3_Report.py` — single-page HTML dump combining `phase1_results.csv` + best thresholds per sweep + Person B's targets. Browser-print to PDF. | — | Stretch |

## Division of Labor

**Pre-session (Person B, ~30 min)**:
- Draft 3 target sets on paper (e.g., conservative: ≥0.85 P, ≥0.70 R; recall-first: ≥0.95 R, ≥0.50 P; balanced: F1 ≥ 0.75).
- Decide on the radio-button label set he actually wants to use (the demo CSV has `off_effort` / `noise` / `beluga`; confirm or expand).
- Skim 5 spectrograms in `data/tuxedni_spectrograms/` so he has a "what good looks like" baseline.
- Verify his clone of the repo is up-to-date and `conda activate bioacustics` works on the VM.

**During session — interleave at the same machine**:
- A codes a slice → B drives that slice on real data → A fixes B's findings → repeat.
- B never reads code; he reads the screen, drives the mouse, and says "wrong term" / "axis labels too small" / "I'd want this filter".
- B owns the demo at t=1:45 — he runs it, A watches.

**Post-session**:
- Merge `frontend-spike` to `active-learning-2nd-round-analysis` only after the demo passes.
- Person B's reviewed CSV under `frontend/reviews/` is what feeds the next AL prep run.

## Critical Files to Read/Reuse Before Coding

- `/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/data/plot_spectrograms.py` — `hz_to_mel`, `CATEGORY_MAP`, `LABEL_COLORS`, `SAMPLE_RATE=24000` (line 29–37).
- `/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/inference/tuxedni_results_1stAL_round_rev.csv` — schema: `file_path, audio, start(s), end(s), pred_label, pred_label_binary, prob_class_0, confidence_binary, pred_label_3class, prob_class_1, prob_class_2, prob_class_3, manual_verif, segment_type`.
- `/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/inference/final_test_round{1,2}_threshold_sweep.csv` — schema: `threshold, beluga_prec, beluga_rec, beluga_f1, orca_f1, nowhale_f1, accuracy, macro_f1`.
- `/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/inference/phase1_e14xr13c_threshold_sweep.csv` — same schema as above.
- `/home/v-druizlopez/v-druizlopez/Bioacoustics/CookInlet_Belugas/al_round2_prepare.py:73-81` — canonical `manual_verif` value set; mirror it in the radio buttons.

## Risk Callouts (ranked by likelihood)

1. **Source `.wav` resolution for audio playback** — the `audio` column holds a basename like `120_00000008.e` (no extension). The actual `.wav` files live somewhere under `/home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/`. **De-risk**: in block 0:00, locate one `.wav` matching one row of the demo CSV. If extension/path discovery takes >10 min, ship without audio for the demo and add it post-session.
2. **Streamlit rerun-per-interaction** — Streamlit re-executes the script on every widget change. Use `@st.cache_data` on `load_predictions` and `@st.cache_resource` on figure objects keyed by row index. Use `st.session_state` for the current row index, not query params.
3. **Concurrent writes** — A and B are on separate clones. Each writes to `frontend/reviews/<input_basename>_<username>_reviewed.csv` (username from `os.environ['USER']`). They merge later by row-key (`file_path` is unique). No locking needed.
4. **`matplotlib.use("Agg")`** — `plot_spectrograms.py` forces Agg at import time. `frontend/spec_render.py` should NOT import that file; copy the 5 lines we need (`hz_to_mel`, `CATEGORY_MAP`, `LABEL_COLORS`) instead. Streamlit + plain matplotlib.figure.Figure works fine.
5. **Branch hygiene** — current branch is `active-learning-2nd-round-analysis` with uncommitted AL Round-2 work. Person A creates `frontend-spike` off it in block 0:00 to keep the spike isolated.
6. **Schema drift across CSVs** — `tuxedni_r2_resting_predictions.csv` has no `manual_verif` column. `csv_io.load_predictions` adds it (default `""`) on first read so the Review page works on any inference output, not just the reviewed file.
7. **CSV size** — demo CSV is 1,095 rows (fine). `tuxedni_r2_resting_predictions.csv` is 565k rows; `csv_io` should warn (not refuse) above 10k rows and recommend filtering before review.

## Verification — Demo Script

At t=2:00, success means Person B drives this end-to-end without A touching the keyboard:

1. Browser at `http://localhost:8501` → home page loads with nav.
2. Sidebar → CSV picker shows `tuxedni_results_1stAL_round_rev.csv` selected by default. Row count "1,095 rows, 47 unverified" displayed.
3. Sidebar → "Review" page. Spectrogram for the first unverified row renders. Probabilities + `pred_label` shown. Audio widget plays a 2-second slice.
4. Click radio button → `beluga`. "Saved at HH:MM:SS" indicator flashes.
5. Click "Next" → next unverified row appears.
6. Click radio button → `noise`. Save flashes.
7. Sidebar → "AL Targets" page. Multi-select 2 sweep CSVs.
8. Set sliders: min_beluga_P=0.85, min_beluga_R=0.70.
9. Verdict per CSV displays: e.g. "round1: best threshold=0.42, beluga F1=0.74" / "round2: no threshold meets target — closest is threshold=0.36 with P=0.81 R=0.70".
10. Open `frontend/reviews/tuxedni_results_1stAL_round_rev_<username>_reviewed.csv` in a fresh terminal → confirm Person B's two new `manual_verif` values are persisted on disk.

**Pass**: all 10 steps work. **Fail-but-ship**: if step 3's audio is silent on that machine but spectrogram + everything else works, ship it and chase the audio path issue post-session.

## Out of Scope (explicitly cut)

- Component #1 (data-prep & training-setup frontend) — Person A handles splits manually; UI saves no time.
- Component #4 (results-synthesis backend) as a real component — its analytical core is covered by AL Targets; the HTML report dump is a stretch goal only.
- User authentication, multi-user real-time sync, deployment beyond `localhost`.
- Triggering training/inference from the UI — long-running CLI ops stay CLI.
- Per-recording-site filters in the demo data (the demo CSV is Tuxedni-only).

## Confidence

**Confidence**: HIGH
- Factual accuracy: HIGH — schemas of the CSVs and helper functions verified by reading the files (`tuxedni_results_1stAL_round_rev.csv` line 1, `final_test_round{1,2}_threshold_sweep.csv` line 1, `plot_spectrograms.py:29-37`).
- Completeness: MEDIUM — source `.wav` path discovery is the one open question; mitigated by the 0:00 smoke test.
- Implementation correctness: MEDIUM — Streamlit code is unwritten; the design is conservative (cached reads, atomic writes) but the demo is the only real test.
