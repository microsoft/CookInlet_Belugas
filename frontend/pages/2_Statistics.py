"""Statistics dashboard: predictions vs. ground truth vs. manual verification.

Shows stage-1 (3-class) and stage-2 (ecotype) confusion matrices and per-class
metrics against either ground truth or re-scored manual verification, plus a
grouped histogram of `prob_Orca` colored by outcome. Two sliders drive the
predictions: a decision threshold on `prob_Orca` (stage 1) and an abstention
threshold on the calibrated max ecotype probability (stage 2).
"""

# pandas stubs declare df[col] as `DataFrame | Series` even when col is a str,
# and altair returns `FacetChart | LayerChart` for layered charts. These trip
# pyright everywhere; the runtime behavior is correct. Suppress at file scope.
# pyright: reportArgumentType=false, reportAssignmentType=false, reportReturnType=false, reportAttributeAccessIssue=false, reportCallIssue=false

from __future__ import annotations

import sys
from pathlib import Path
from typing import cast

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from branding import render_logo

st.set_page_config(page_title="SPIRAL · Statistics", page_icon="📊", layout="wide")


# ── Bootstrap ────────────────────────────────────────────────────────────────

render_logo()

if "df" not in st.session_state:
    st.warning("Load a CSV from the home page sidebar first.")
    st.stop()

_df_raw: pd.DataFrame = st.session_state["df"]


# ── Schema introspection ─────────────────────────────────────────────────────

PRED_COL: str = config.PRED_LABEL_COLUMN
PRED_LABELS: dict[int, str] = config.PRED_LABELS
KW_PRED_VALUES = set(getattr(config, "OUTCOME_POSITIVE_PRED_VALUES", set()) or set())
MANUAL_3CLASS_COL: str = config.MANUAL_VERIF_COLUMN
MANUAL_ECOTYPE_COL: str | None = getattr(config, "MANUAL_VERIF_STAGE2_COLUMN", None)

# GT specs from config (column name, int→name map), bucketed by stage.
_GT_3CLASS: tuple[str, dict[int, str]] | None = None
_GT_ECOTYPE: tuple[str, dict[int, str]] | None = None
for _col, _, _vmap in getattr(config, "GROUND_TRUTH_COLUMNS", []):
    if "3class" in _col.lower():
        _GT_3CLASS = (_col, _vmap)
    elif "ecotype" in _col.lower():
        _GT_ECOTYPE = (_col, _vmap)

# Cascade integer → ecotype name (subset of PRED_LABELS that are ecotype classes).
ECOTYPE_PRED_INTS: dict[int, str] = {
    i: PRED_LABELS[i]
    for i in KW_PRED_VALUES
    if PRED_LABELS.get(i) not in ("Unassigned_KW", None)
}

# 3-class fold for the cascade prediction (8-class → {NonBio, Bio, Orca}).
PRED_TO_3CLASS: dict[int, str] = {0: "NonBio", 1: "Bio"}
for _i in KW_PRED_VALUES:
    PRED_TO_3CLASS[_i] = "Orca"


# ── Enrichment ───────────────────────────────────────────────────────────────


def _series(df: pd.DataFrame, col: str) -> pd.Series:
    """`df[col]` narrowed to a Series for the type checker."""
    return cast(pd.Series, df[col])


def _map_int_to_str(series: pd.Series, mapping: dict[int, str]) -> pd.Series:
    """Vectorized int-to-str map that survives NA values."""
    return series.apply(
        lambda v: mapping.get(int(v)) if pd.notna(v) else None  # type: ignore[arg-type]
    )


def _enrich(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns used by every plot: folded pred labels and GT names."""
    out = df.copy()
    pred_int = _series(out, PRED_COL).astype("Int64")
    out["_pred_3class"] = _map_int_to_str(pred_int, PRED_TO_3CLASS)
    out["_pred_ecotype"] = _map_int_to_str(pred_int, ECOTYPE_PRED_INTS)
    if _GT_3CLASS and _GT_3CLASS[0] in out.columns:
        out["_gt_3class"] = _map_int_to_str(
            _series(out, _GT_3CLASS[0]).astype("Int64"), _GT_3CLASS[1]
        )
    if _GT_ECOTYPE and _GT_ECOTYPE[0] in out.columns:
        out["_gt_ecotype"] = _map_int_to_str(
            _series(out, _GT_ECOTYPE[0]).astype("Int64"), _GT_ECOTYPE[1]
        )
    return out


df = _enrich(_df_raw)


df_view = df


# ── Plot 1: Confusion matrices ───────────────────────────────────────────────


def _cm_chart(
    sub: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    *,
    title: str,
    class_order: list[str] | None = None,
):
    """Build an Altair count-heatmap from two categorical columns of `sub`."""
    cm = pd.crosstab(sub[pred_col], sub[ref_col], dropna=False)
    tidy = cm.reset_index().melt(
        id_vars=pred_col, var_name=ref_col, value_name="count"
    )
    tidy["count_label"] = tidy["count"].astype(int).astype(str)
    max_count = int(tidy["count"].max()) if len(tidy) else 0
    x_sort = class_order if class_order else alt.Undefined
    y_sort = class_order if class_order else alt.Undefined
    base = alt.Chart(tidy).encode(
        x=alt.X(f"{ref_col}:N", title="Truth", sort=x_sort),
        y=alt.Y(f"{pred_col}:N", title="Prediction", sort=y_sort),
    )
    heat = base.mark_rect().encode(
        color=alt.Color(
            "count:Q", title="Count", scale=alt.Scale(scheme="blues")
        ),
        tooltip=[
            alt.Tooltip(f"{pred_col}:N", title="Prediction"),
            alt.Tooltip(f"{ref_col}:N", title="Truth"),
            alt.Tooltip("count:Q", title="Count", format=","),
        ],
    )
    threshold = max(max_count / 2, 1)
    text = base.mark_text(baseline="middle").encode(
        text=alt.Text("count_label:N"),
        color=alt.condition(
            f"datum.count > {threshold}",
            alt.value("white"),
            alt.value("black"),
        ),
    )
    return (heat + text).properties(title=title, height=380)


# Manual stage-1 label → 3-class label. "Humpback" folds into Bio (the taxonomy
# bucket). "Unsure" is excluded from the rescored truth (Section 4 of the
# pipeline review report).
_MANUAL_TO_3CLASS: dict[str, str] = {
    "NonBio": "NonBio",
    "Bio": "Bio",
    "Humpback": "Bio",
    "Orca": "Orca",
}


def _rescored_3class_truth(df: pd.DataFrame) -> pd.Series:
    """Effective 3-class truth: annotation, overridden by manual where set.

    Implements the re-score from Section 4 of complete_pipeline_review.md:
      NonBio   → NonBio
      Bio      → Bio
      Humpback → Bio   (taxonomy fold)
      Orca     → Orca
      Unsure   → excluded (truth set to NA, row dropped from metrics)
    Where the reviewer did not look, the GT annotation is kept as truth.
    """
    if "_gt_3class" in df.columns:
        truth = _series(df, "_gt_3class").copy().astype("object")
    else:
        truth = pd.Series([None] * len(df), index=df.index, dtype="object")
    if MANUAL_3CLASS_COL not in df.columns:
        return truth
    mv = _series(df, MANUAL_3CLASS_COL).fillna("").astype(str).str.strip()
    has_label = mv.isin(list(_MANUAL_TO_3CLASS.keys()))
    truth.loc[has_label] = mv[has_label].map(_MANUAL_TO_3CLASS)
    truth.loc[mv.eq("Unsure")] = None
    return truth


def _rescored_ecotype_truth(df: pd.DataFrame, classes: list[str]) -> pd.Series:
    """Effective ecotype truth: annotation, overridden by manual where set.

    Mirrors the 3-class re-score (Section 4):
      SRKW / TKW / SAR / NRKW / OKW → that class (direct)
      Unassigned                    → excluded (truth NA)
      Unsure                        → excluded (truth NA)
    Where the reviewer did not look, the GT ecotype annotation is kept; the
    sentinel "—" (no ecotype assignment) is excluded.
    """
    if "_gt_ecotype" in df.columns:
        truth = _series(df, "_gt_ecotype").copy().astype("object")
        truth = truth.where(truth.astype(str) != "—")
    else:
        truth = pd.Series([None] * len(df), index=df.index, dtype="object")
    if not MANUAL_ECOTYPE_COL or MANUAL_ECOTYPE_COL not in df.columns:
        return truth
    mv = _series(df, MANUAL_ECOTYPE_COL).fillna("").astype(str).str.strip()
    has_label = mv.isin(classes)
    truth.loc[has_label] = mv[has_label]
    truth.loc[mv.isin(["Unassigned", "Unsure"])] = None
    return truth


def _per_class_metrics(
    pred: pd.Series, truth: pd.Series, classes: list[str]
) -> pd.DataFrame:
    """Precision / recall / F1 / support per class, plus macro avg.

    Computed over rows where both `pred` and `truth` are non-null. Macro avg
    averages precision/recall/F1 across the listed classes (sklearn-style).
    """
    mask = pred.notna() & truth.notna()
    p = pred[mask].astype(str)
    t = truth[mask].astype(str)
    rows = []
    for cls in classes:
        tp = int(((p == cls) & (t == cls)).sum())
        fp = int(((p == cls) & (t != cls)).sum())
        fn = int(((p != cls) & (t == cls)).sum())
        support = int((t == cls).sum())
        precision = tp / (tp + fp) if (tp + fp) else float("nan")
        recall = tp / (tp + fn) if (tp + fn) else float("nan")
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision and recall and (precision + recall) > 0
            else float("nan")
        )
        rows.append(
            {
                "class": cls,
                "precision": round(precision, 3) if not np.isnan(precision) else None,
                "recall": round(recall, 3) if not np.isnan(recall) else None,
                "F1": round(f1, 3) if not np.isnan(f1) else None,
                "support": support,
            }
        )
    out = pd.DataFrame(rows)
    if len(out):
        macro = {
            "class": "macro avg",
            "precision": round(out["precision"].astype(float).mean(skipna=True), 3),
            "recall": round(out["recall"].astype(float).mean(skipna=True), 3),
            "F1": round(out["F1"].astype(float).mean(skipna=True), 3),
            "support": int(out["support"].sum()),
        }
        out.loc[len(out)] = macro
    return out


def _threshold_pred_3class(df: pd.DataFrame, threshold: float) -> pd.Series | None:
    """Threshold-derived stage-1 prediction.

    Orca when `prob_Orca >= threshold`; otherwise argmax of (NonBio, Bio).
    Returns None if the required probability columns are missing.
    """
    need = ["prob_NonBio", "prob_Bio", "prob_Orca"]
    if not all(c in df.columns for c in need):
        return None
    p_nonbio = _series(df, "prob_NonBio").astype(float)
    p_bio = _series(df, "prob_Bio").astype(float)
    p_orca = _series(df, "prob_Orca").astype(float)
    is_orca = p_orca >= threshold
    nonbio_higher = p_nonbio >= p_bio
    pred = np.where(is_orca, "Orca", np.where(nonbio_higher, "NonBio", "Bio"))
    return pd.Series(pred, index=df.index, dtype="object")


def _threshold_pred_ecotype(
    df: pd.DataFrame,
    orca_threshold: float,
    unassigned_threshold: float,
    classes: list[str],
) -> pd.Series | None:
    """Threshold-derived stage-2 prediction.

    For rows where stage-1 says Orca (`prob_Orca >= orca_threshold`):
      * if `max(prob_<class>) >= unassigned_threshold`, predict the argmax class
      * otherwise predict `"Unassigned"` (the model is unsure which ecotype).
    Rows where stage-1 does not say Orca are excluded (None). Returns None if
    any required probability column is missing.
    """
    if "prob_Orca" not in df.columns:
        return None
    eco_cols = [f"prob_{c}" for c in classes]
    if not all(c in df.columns for c in eco_cols):
        return None
    p_orca = _series(df, "prob_Orca").astype(float)
    is_orca = p_orca >= orca_threshold
    eco_probs = np.asarray(df[eco_cols].astype(float).values)
    eco_max = eco_probs.max(axis=1)
    argmax_idx = eco_probs.argmax(axis=1)
    is_assigned = eco_max >= unassigned_threshold
    pred = pd.Series([None] * len(df), index=df.index, dtype="object")
    mask_assigned = is_orca & is_assigned
    if mask_assigned.any():
        pred.loc[mask_assigned] = [
            classes[i] for i in argmax_idx[mask_assigned.to_numpy()]
        ]
    pred.loc[is_orca & ~is_assigned] = "Unassigned"
    return pred


def _optimal_orca_f1_threshold(
    df: pd.DataFrame, truth: pd.Series | None
) -> float:
    """Sweep prob_Orca thresholds and return the one that maximises F1 for
    Orca-vs-rest against `truth` (a 3-class string Series). Returns 0.5 if the
    inputs don't allow a meaningful computation."""
    if truth is None or "prob_Orca" not in df.columns:
        return 0.5
    p_orca = _series(df, "prob_Orca").astype(float)
    mask = p_orca.notna() & truth.notna()
    if int(mask.sum()) == 0:
        return 0.5
    score = np.asarray(p_orca[mask].to_numpy(), dtype=float)
    ref_pos = np.asarray(truth[mask].astype(str).eq("Orca").to_numpy())
    if not ref_pos.any():
        return 0.5
    best_f1 = -1.0
    best_thr = 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        pred_pos = score >= thr
        tp = int((pred_pos & ref_pos).sum())
        fp = int((pred_pos & ~ref_pos).sum())
        fn = int((~pred_pos & ref_pos).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return round(best_thr, 2)


def _render_cm_block(
    sub: pd.DataFrame,
    pred_col: str,
    ref_col: str,
    classes: list[str],
    *,
    title: str,
    caption: str | None = None,
    extra_pred_classes: list[str] | None = None,
) -> None:
    """Render confusion matrix (wide, left) + per-class metrics (right).

    `extra_pred_classes` adds rows to the prediction axis ordering only (they
    don't get a per-class metric row). Useful for catch-all buckets like
    "Unassigned" which appear as predictions but not as truth.
    """
    chart_order = classes + (extra_pred_classes or [])
    col_cm, col_metrics = st.columns([2, 1])
    with col_cm:
        st.altair_chart(
            _cm_chart(sub, pred_col, ref_col, title=title, class_order=chart_order),
            use_container_width=True,
        )
    with col_metrics:
        st.markdown("**Per-class metrics**")
        metrics_df = _per_class_metrics(sub[pred_col], sub[ref_col], classes)
        st.dataframe(metrics_df, hide_index=True, use_container_width=True)
        if caption:
            st.caption(caption)


# ── Page body ────────────────────────────────────────────────────────────────

st.caption(
    "Compare the cascade prediction against the chosen reference. "
    "Cell values are raw counts; per-class precision / recall / F1 / support "
    "are computed alongside. Move the threshold sliders to see how the "
    "matrices and metrics shift."
)
ref_choice = st.radio(
    "Reference",
    ["Ground truth", "Manual verification"],
    captions=[
        "The annotation as recorded in the CSV.",
        "Re-scored truth — the reviewer's label overrides the annotation where "
        "set; rows where the reviewer did not look keep the annotation as truth.",
    ],
    horizontal=True,
    key="cm_ref",
)

classes_3c = ["NonBio", "Bio", "Orca"]
classes_ec = list(ECOTYPE_PRED_INTS.values())

# ── Truths (depend on the reference toggle) ──────────────────────────────────
if ref_choice == "Ground truth":
    ref_label = "GT 3-class"
    s1_truth = _series(df_view, "_gt_3class") if "_gt_3class" in df_view.columns else None
    s1_caption = "Truth = `label_3class` annotation."
    if "_gt_ecotype" in df_view.columns:
        s2_truth = _series(df_view, "_gt_ecotype")
        s2_truth = s2_truth.where(s2_truth.astype(str) != "—")
        s2_truth = s2_truth.where(s2_truth.notna() & s2_truth.astype(str).ne(""))
    else:
        s2_truth = None
    s2_ref_label = "GT ecotype"
    s2_caption = "Truth = `ecotype_label` annotation; rows without an ecotype label excluded."
else:
    ref_label = "manual-rescored 3-class"
    s1_truth = _rescored_3class_truth(df_view)
    if MANUAL_3CLASS_COL in df_view.columns:
        mv = _series(df_view, MANUAL_3CLASS_COL).fillna("").astype(str).str.strip()
        n_overrides = int(mv.isin(list(_MANUAL_TO_3CLASS.keys())).sum())
        n_unsure = int(mv.eq("Unsure").sum())
        s1_caption = (
            f"Truth = annotation, overridden by reviewer on **{n_overrides:,}** rows "
            f"(Humpback→Bio); **{n_unsure:,}** Unsure rows excluded."
        )
    else:
        s1_caption = "Truth = `label_3class` annotation (no manual column present)."
    s2_truth = _rescored_ecotype_truth(df_view, classes_ec)
    n_overrides_ec = 0
    n_excluded_ec = 0
    if MANUAL_ECOTYPE_COL and MANUAL_ECOTYPE_COL in df_view.columns:
        mv_ec = _series(df_view, MANUAL_ECOTYPE_COL).fillna("").astype(str).str.strip()
        n_overrides_ec = int(mv_ec.isin(classes_ec).sum())
        n_excluded_ec = int(mv_ec.isin(["Unassigned", "Unsure"]).sum())
    s2_ref_label = "manual-rescored ecotype"
    s2_caption = (
        f"Truth = annotation, overridden by reviewer on **{n_overrides_ec:,}** rows; "
        f"**{n_excluded_ec:,}** Unassigned/Unsure rows excluded."
    )

# ── Threshold (default = Orca-F1 optimum vs current reference) ───────────────
optimal_thr = _optimal_orca_f1_threshold(df_view, s1_truth)
threshold = st.slider(
    "Decision threshold (prob_Orca)",
    min_value=0.0,
    max_value=1.0,
    value=optimal_thr,
    step=0.01,
    key="cm_threshold",
    help=(
        "Stage 1 prediction: Orca if `prob_Orca ≥ threshold`, else argmax of "
        "(prob_NonBio, prob_Bio). The default is the threshold that maximises "
        "Orca F1 against the current reference."
    ),
)
st.caption(
    f"💡 Threshold maximising Orca F1 vs the current reference: "
    f"**{optimal_thr:.2f}**"
)

pred_3c_thr = _threshold_pred_3class(df_view, threshold)
if pred_3c_thr is None:
    st.warning(
        "Stage-1 probability columns missing (prob_NonBio / prob_Bio / prob_Orca) "
        "— falling back to the cascade's discrete prediction; the threshold "
        "slider has no effect for stage 1."
    )
    pred_3c_thr = _series(df_view, "_pred_3class") if "_pred_3class" in df_view.columns else None

# ── Orca probability distribution by outcome ─────────────────────────────────
if "prob_Orca" in df_view.columns and s1_truth is not None:
    st.markdown("### Orca probability distribution by outcome")
    st.caption(
        f"Binary view (Orca vs not-Orca) at threshold = **{threshold:.2f}** "
        "(dashed line). Bars are stacked by outcome (TP / FP / FN / TN) on a "
        "log Y axis."
    )
    score = _series(df_view, "prob_Orca").astype(float).clip(0.0, 1.0)
    mask_hist = score.notna() & s1_truth.notna()
    if int(mask_hist.sum()) == 0:
        st.info("No rows with both prob_Orca and the chosen reference.")
    else:
        s_arr = np.asarray(score[mask_hist].to_numpy())
        ref_pos = np.asarray(s1_truth[mask_hist].astype(str).eq("Orca").to_numpy())
        pred_pos = s_arr >= threshold
        outcome = np.where(
            pred_pos & ref_pos, "TP",
            np.where(pred_pos & ~ref_pos, "FP",
                np.where(~pred_pos & ref_pos, "FN", "TN")
            ),
        )
        # Pre-aggregate with explicit bin_start/bin_end so Altair gives each
        # bar its bin's full width on the x-axis (a quantitative `x` alone
        # produces 1-pixel marks). Sort by count descending so the largest bar
        # draws FIRST (behind) and the smallest draws LAST (on top), which
        # gives a "stacked"-looking visual without needing transparency.
        n_bins = 30
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_width = 1.0 / n_bins
        bin_idx = np.clip(np.digitize(s_arr, edges) - 1, 0, n_bins - 1)
        # Shrink each bar's x-extent slightly so adjacent bins show a gap and
        # the chart reads as discrete bars rather than a filled area.
        bar_pad = bin_width * 0.15
        agg = (
            pd.DataFrame(
                {
                    "bin_start": edges[bin_idx] + bar_pad,
                    "bin_end": edges[bin_idx + 1] - bar_pad,
                    "outcome": outcome,
                }
            )
            .groupby(["bin_start", "bin_end", "outcome"], sort=False)
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            .reset_index(drop=True)
        )
        # Explicit y baseline at 1 so bars render as rectangles on the log
        # scale — without it Vega-Lite has no defined bottom (log(0) = -∞) and
        # falls back to drawing each mark as a 1-pixel horizontal stripe.
        agg["count_floor"] = 1
        outcome_colors = {"TP": "#2ca02c", "FP": "#d62728", "FN": "#ff7f0e", "TN": "#1f77b4"}
        hist = (
            alt.Chart(agg)
            .mark_bar()
            .encode(
                x=alt.X(
                    "bin_start:Q",
                    title="prob_Orca",
                    scale=alt.Scale(domain=[0.0, 1.0]),
                ),
                x2="bin_end:Q",
                y=alt.Y(
                    "count:Q",
                    title="Rows (log)",
                    scale=alt.Scale(type="log"),
                    stack=None,
                ),
                y2="count_floor:Q",
                color=alt.Color(
                    "outcome:N",
                    title="Outcome",
                    scale=alt.Scale(
                        domain=list(outcome_colors.keys()),
                        range=list(outcome_colors.values()),
                    ),
                    sort=["TP", "FP", "FN", "TN"],
                ),
                order=alt.Order("count:Q", sort="descending"),
                tooltip=[
                    alt.Tooltip("outcome:N", title="Outcome"),
                    alt.Tooltip("bin_start:Q", title="prob_Orca bin start", format=".3f"),
                    alt.Tooltip("count:Q", title="Rows", format=","),
                ],
            )
        )
        thr_rule = (
            alt.Chart(pd.DataFrame({"thr": [threshold]}))
            .mark_rule(strokeDash=[4, 4], color="black")
            .encode(x=alt.X("thr:Q", scale=alt.Scale(domain=[0.0, 1.0])))
        )
        st.altair_chart((hist + thr_rule).properties(height=320), use_container_width=True)

# ── Stage 1 (3-class) ────────────────────────────────────────────────────────
st.markdown("### Stage 1 (3-class)")
if s1_truth is None or pred_3c_thr is None or int(s1_truth.notna().sum()) == 0:
    if ref_choice == "Manual verification":
        st.info(
            "No re-scored truth available — verify some rows manually or "
            "ensure the CSV has a `label_3class` column."
        )
    else:
        st.info("No GT 3-class column in this CSV.")
else:
    sub = df_view.assign(_pred_thr_3c=pred_3c_thr, _ref_3class=s1_truth).dropna(
        subset=["_pred_thr_3c", "_ref_3class"]
    )
    if sub.empty:
        st.info("No rows with both prediction and the chosen reference for stage 1.")
    else:
        _render_cm_block(
            sub,
            "_pred_thr_3c",
            "_ref_3class",
            classes_3c,
            title=f"Stage 1: pred × {ref_label}  (n = {len(sub):,})",
            caption=s1_caption,
        )

# ── Stage 2 (ecotype) ────────────────────────────────────────────────────────
st.markdown("### Stage 2 (ecotype)")
st.caption(
    "Only rows where stage 1 predicts Orca (at the current threshold) appear "
    "on the prediction axis. The slider below is the **ecotype abstention "
    "threshold** — the pipeline's stage-2 knob: if the calibrated "
    "max-confidence over the 5 ecotypes is below it, the row is emitted as "
    "Unassigned instead of being assigned to the argmax ecotype."
)
_default_abst = float(getattr(config, "ECOTYPE_ABSTENTION_THRESHOLD", 0.94))
unassigned_thr = st.slider(
    "Ecotype 'Unassigned' threshold",
    min_value=0.0,
    max_value=1.0,
    value=_default_abst,
    step=0.01,
    key="cm_unassigned_threshold",
    help=(
        f"Pipeline default = **{_default_abst:.4f}** (from "
        "`data_config.yaml::cascade.threshold`). Move the slider to see how "
        "the stage-2 confusion matrix and per-class metrics shift when the "
        "abstention floor changes."
    ),
)

pred_ec_thr = _threshold_pred_ecotype(df_view, threshold, unassigned_thr, classes_ec)
if pred_ec_thr is None:
    pred_ec_thr = _series(df_view, "_pred_ecotype") if "_pred_ecotype" in df_view.columns else None

if s2_truth is None or pred_ec_thr is None or int(s2_truth.notna().sum()) == 0:
    st.info(f"No rows with {s2_ref_label} available.")
else:
    sub2 = df_view.assign(_pred_thr_ec=pred_ec_thr, _ref_ecotype=s2_truth).dropna(
        subset=["_pred_thr_ec", "_ref_ecotype"]
    )
    if sub2.empty:
        st.info("No ecotype-predicted rows with the chosen reference.")
    else:
        _render_cm_block(
            sub2,
            "_pred_thr_ec",
            "_ref_ecotype",
            classes_ec,
            title=f"Stage 2: pred × {s2_ref_label}  (n = {len(sub2):,})",
            caption=s2_caption,
            extra_pred_classes=["Unassigned"],
        )

