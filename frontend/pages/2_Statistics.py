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
    tidy = cm.reset_index().melt(id_vars=pred_col, var_name=ref_col, value_name="count")
    tidy["count_label"] = tidy["count"].astype(int).astype(str)
    max_count = int(tidy["count"].max()) if len(tidy) else 0
    x_sort = class_order if class_order else alt.Undefined
    y_sort = class_order if class_order else alt.Undefined
    base = alt.Chart(tidy).encode(
        x=alt.X(f"{ref_col}:N", title="Truth", sort=x_sort),
        y=alt.Y(f"{pred_col}:N", title="Prediction", sort=y_sort),
    )
    heat = base.mark_rect().encode(
        color=alt.Color("count:Q", title="Count", scale=alt.Scale(scheme="blues")),
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


_THREE_CLASSES = ["NonBio", "Bio", "Orca"]


def _threshold_pred_3class(
    df: pd.DataFrame, target: str, threshold: float
) -> pd.Series | None:
    """Threshold-derived stage-1 prediction.

    Predict `target` when `prob_<target> >= threshold`; otherwise predict the
    argmax over the remaining two classes' probabilities. Returns None if any
    required probability column is missing.
    """
    need = [f"prob_{c}" for c in _THREE_CLASSES]
    if not all(c in df.columns for c in need):
        return None
    others = [c for c in _THREE_CLASSES if c != target]
    p_target = _series(df, f"prob_{target}").astype(float)
    p_other = np.asarray(df[[f"prob_{c}" for c in others]].astype(float).values)
    is_target = p_target >= threshold
    other_argmax_idx = p_other.argmax(axis=1)
    other_argmax = np.asarray([others[i] for i in other_argmax_idx])
    pred = np.where(is_target, target, other_argmax)
    return pd.Series(pred, index=df.index, dtype="object")


def _threshold_pred_ecotype(
    df: pd.DataFrame,
    stage1_pred: pd.Series | None,
    unassigned_threshold: float,
    classes: list[str],
) -> pd.Series | None:
    """Threshold-derived stage-2 prediction.

    For rows where the stage-1 prediction is `"Orca"`:
      * if `max(prob_<class>) >= unassigned_threshold`, predict the argmax class
      * otherwise predict `"Unassigned"` (the model is unsure which ecotype).
    Rows where stage-1 did not predict Orca are excluded (None). Returns None
    if `stage1_pred` is missing or any required probability column is missing.
    """
    if stage1_pred is None:
        return None
    eco_cols = [f"prob_{c}" for c in classes]
    if not all(c in df.columns for c in eco_cols):
        return None
    is_orca = stage1_pred.astype(str).eq("Orca")
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


# Threshold sweep used by Fβ-optimum and PR-curve helpers. Coarse 0.01 grid
# across [0.01, 0.99] plus a finer 0.001 grid in [0.991, 0.999] so the curve
# can show recall behaviour at the very top of the prob_Orca distribution
# (most truly-positive rows cluster near 1.0, so recall barely moves until
# you cross 0.99 — the fine tail makes that region visible).
_THRESHOLD_SWEEP = np.concatenate(
    [np.linspace(0.01, 0.99, 99), np.linspace(0.991, 0.999, 9)]
)


def _fbeta(prec: float, rec: float, beta: float) -> float:
    """Standard Fβ score: (1+β²) · P · R / (β² · P + R)."""
    b2 = beta * beta
    denom = b2 * prec + rec
    if denom <= 0:
        return 0.0
    return (1.0 + b2) * prec * rec / denom


def _optimal_class_fb_threshold(
    df: pd.DataFrame,
    target: str,
    truth: pd.Series | None,
    beta: float,
) -> float:
    """Sweep `prob_<target>` and return the threshold that maximises Fβ for
    target-vs-rest against `truth` (a 3-class string Series).

    Returns 0.5 if the inputs don't allow a meaningful computation.
    """
    col = f"prob_{target}"
    if truth is None or col not in df.columns:
        return 0.5
    p = _series(df, col).astype(float)
    mask = p.notna() & truth.notna()
    if int(mask.sum()) == 0:
        return 0.5
    score = np.asarray(p[mask].to_numpy(), dtype=float)
    ref_pos = np.asarray(truth[mask].astype(str).eq(target).to_numpy())
    if not ref_pos.any():
        return 0.5
    best = -1.0
    best_thr = 0.5
    for thr in _THRESHOLD_SWEEP:
        pred_pos = score >= thr
        tp = int((pred_pos & ref_pos).sum())
        fp = int((pred_pos & ~ref_pos).sum())
        fn = int((~pred_pos & ref_pos).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f = _fbeta(prec, rec, beta)
        if f > best:
            best = f
            best_thr = float(thr)
    return round(best_thr, 2)


def _optimal_ecotype_macro_fb_threshold(
    df: pd.DataFrame,
    stage1_pred: pd.Series | None,
    truth: pd.Series | None,
    classes: list[str],
    beta: float,
) -> float | None:
    """Sweep the abstention threshold and return the one that maximises macro
    Fβ across the 5 ecotype classes (Unassigned excluded — it's not a target
    label in `truth`). Returns None if the inputs don't allow it.
    """
    if stage1_pred is None or truth is None:
        return None
    eco_cols = [f"prob_{c}" for c in classes]
    if not all(c in df.columns for c in eco_cols):
        return None
    is_orca = stage1_pred.astype(str).eq("Orca").to_numpy()
    truth_arr = truth.fillna("").astype(str).to_numpy()
    valid = truth_arr != ""
    if not (is_orca & valid).any():
        return None
    eco_probs = np.asarray(df[eco_cols].astype(float).values)
    eco_max = eco_probs.max(axis=1)
    argmax_idx = eco_probs.argmax(axis=1)
    argmax_class = np.asarray([classes[i] for i in argmax_idx])
    best = -1.0
    best_thr = float(getattr(config, "ECOTYPE_ABSTENTION_THRESHOLD", 0.94))
    for thr in _THRESHOLD_SWEEP:
        is_assigned = eco_max >= thr
        pred = np.full(len(df), "", dtype=object)
        pred[is_orca & is_assigned] = argmax_class[is_orca & is_assigned]
        pred[is_orca & ~is_assigned] = "Unassigned"
        f_scores = []
        for cls in classes:
            tp = int(((pred == cls) & (truth_arr == cls) & valid).sum())
            fp = int(((pred == cls) & (truth_arr != cls) & valid).sum())
            fn = int(((pred != cls) & (truth_arr == cls) & valid).sum())
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f_scores.append(_fbeta(prec, rec, beta))
        macro = float(np.mean(f_scores))
        if macro > best:
            best = macro
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
    s1_truth = (
        _series(df_view, "_gt_3class") if "_gt_3class" in df_view.columns else None
    )
    s1_caption = "Truth = `label_3class` annotation."
    if "_gt_ecotype" in df_view.columns:
        s2_truth = _series(df_view, "_gt_ecotype")
        s2_truth = s2_truth.where(s2_truth.astype(str) != "—")
        s2_truth = s2_truth.where(s2_truth.notna() & s2_truth.astype(str).ne(""))
    else:
        s2_truth = None
    s2_ref_label = "GT ecotype"
    s2_caption = (
        "Truth = `ecotype_label` annotation; rows without an ecotype label excluded."
    )
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

# ── Optimization target (left) + PR curve (right) ────────────────────────────
opt_left, opt_right = st.columns(2)

with opt_left:
    st.markdown("#### Optimization target")
    target_class = st.radio(
        "Target class",
        _THREE_CLASSES,
        index=_THREE_CLASSES.index("Orca"),
        horizontal=True,
        key="opt_target_class",
        help=(
            "Which stage-1 class the decision threshold gates. The slider "
            "below thresholds `prob_<target>`; rows above the threshold are "
            "predicted as the target, rows below get argmax of the remaining "
            "two classes."
        ),
    )
    objective = st.radio(
        "Objective",
        ["Balanced", "Catch every detection", "Be confident"],
        captions=[
            "F1 — equal weight on precision and recall.",
            "F2 — favours recall (fewer missed detections).",
            "F0.5 — favours precision (fewer false alarms).",
        ],
        horizontal=True,
        key="opt_objective",
    )

_OBJECTIVE_TO_BETA = {
    "Balanced": 1.0,
    "Catch every detection": 2.0,
    "Be confident": 0.5,
}
beta = _OBJECTIVE_TO_BETA[objective]
beta_label = "F1" if beta == 1.0 else (f"F{int(beta)}" if beta >= 1 else f"F{beta:g}")
optimal_thr = _optimal_class_fb_threshold(df_view, target_class, s1_truth, beta)
slider_key = f"cm_threshold_{target_class}_b{beta}"
# The slider lives below this two-column block. Read its current value from
# session_state so the PR-curve "current" marker tracks live slider moves.
# Falls back to the Fβ optimum on the first render of a new (target, β) pair.
current_thr_for_pr = float(st.session_state.get(slider_key, optimal_thr))

with opt_right:
    st.markdown(f"#### {target_class} precision-recall curve")
    _score_col = f"prob_{target_class}"
    if _score_col not in df_view.columns or s1_truth is None:
        st.info("No probability column or reference available for the PR curve.")
    else:
        _pr_p = _series(df_view, _score_col).astype(float)
        _pr_mask = _pr_p.notna() & s1_truth.notna()
        if int(_pr_mask.sum()) == 0:
            st.info("No rows with both score and reference for the PR curve.")
        else:
            _pr_score = np.asarray(_pr_p[_pr_mask].to_numpy(), dtype=float)
            _pr_ref_pos = np.asarray(
                s1_truth[_pr_mask].astype(str).eq(target_class).to_numpy()
            )
            if not _pr_ref_pos.any():
                st.info(
                    f"No {target_class} samples in the reference — cannot draw a PR curve."
                )
            else:
                _rows = []
                for _thr in _THRESHOLD_SWEEP:
                    _pred_pos = _pr_score >= _thr
                    _tp = int((_pred_pos & _pr_ref_pos).sum())
                    _fp = int((_pred_pos & ~_pr_ref_pos).sum())
                    _fn = int((~_pred_pos & _pr_ref_pos).sum())
                    _prec = _tp / (_tp + _fp) if (_tp + _fp) else float("nan")
                    _rec = _tp / (_tp + _fn) if (_tp + _fn) else 0.0
                    _rows.append(
                        {"threshold": float(_thr), "precision": _prec, "recall": _rec}
                    )
                pr_df = pd.DataFrame(_rows).dropna(subset=["precision"])
                # Close the curve at both ends, following the sklearn convention:
                #   threshold = 0.0 → predict everything → recall = 1,
                #     precision = prevalence (positives / total).
                #   threshold = 1.0 → predict nothing → recall = 0,
                #     precision is defined as 1 (vacuously perfect).
                _prevalence = float(_pr_ref_pos.mean())
                pr_df = pd.concat(
                    [
                        pd.DataFrame(
                            [
                                {
                                    "threshold": 0.0,
                                    "precision": _prevalence,
                                    "recall": 1.0,
                                }
                            ]
                        ),
                        pr_df,
                        pd.DataFrame(
                            [{"threshold": 1.0, "precision": 1.0, "recall": 0.0}]
                        ),
                    ],
                    ignore_index=True,
                )
                current_row = pr_df.iloc[
                    (pr_df["threshold"] - current_thr_for_pr).abs().argmin()
                ]
                optimal_row = pr_df.iloc[
                    (pr_df["threshold"] - optimal_thr).abs().argmin()
                ]
                marker_df = pd.DataFrame(
                    [
                        {**current_row.to_dict(), "kind": "current"},
                        {**optimal_row.to_dict(), "kind": "optimum"},
                    ]
                )
                curve = (
                    alt.Chart(pr_df)
                    .mark_line(color="steelblue", strokeWidth=2)
                    .encode(
                        x=alt.X(
                            "recall:Q",
                            title="Recall",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        y=alt.Y(
                            "precision:Q",
                            title="Precision",
                            scale=alt.Scale(domain=[0, 1]),
                        ),
                        tooltip=[
                            alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
                            alt.Tooltip("precision:Q", title="Precision", format=".3f"),
                            alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                        ],
                    )
                )
                markers = (
                    alt.Chart(marker_df)
                    .mark_point(size=180, filled=True)
                    .encode(
                        x="recall:Q",
                        y="precision:Q",
                        color=alt.Color(
                            "kind:N",
                            title=None,
                            scale=alt.Scale(
                                domain=["current", "optimum"],
                                range=["#d62728", "#2ca02c"],
                            ),
                            legend=alt.Legend(orient="bottom-left"),
                        ),
                        tooltip=[
                            alt.Tooltip("kind:N", title="Marker"),
                            alt.Tooltip("threshold:Q", title="Threshold", format=".2f"),
                            alt.Tooltip("precision:Q", title="Precision", format=".3f"),
                            alt.Tooltip("recall:Q", title="Recall", format=".3f"),
                        ],
                    )
                )
                st.altair_chart(
                    (curve + markers).properties(height=320),
                    use_container_width=True,
                )

# ── Decision threshold slider (full width, below the side-by-side block) ─────
threshold = st.slider(
    f"Decision threshold (prob_{target_class})",
    min_value=0.0,
    max_value=1.0,
    value=optimal_thr,
    step=0.01,
    key=slider_key,
    help=(
        f"Stage 1 prediction: {target_class} if `prob_{target_class} ≥ threshold`, "
        f"else argmax of the other two classes. The default is the threshold that "
        f"maximises {target_class} {beta_label} against the current reference."
    ),
)
st.caption(
    f"💡 Threshold maximising **{target_class} {beta_label}** "
    f"({objective.lower()}) vs the current reference: **{optimal_thr:.2f}**"
)

pred_3c_thr = _threshold_pred_3class(df_view, target_class, threshold)
if pred_3c_thr is None:
    st.warning(
        "Stage-1 probability columns missing (prob_NonBio / prob_Bio / prob_Orca) "
        "— falling back to the cascade's discrete prediction; the threshold "
        "slider has no effect for stage 1."
    )
    pred_3c_thr = (
        _series(df_view, "_pred_3class") if "_pred_3class" in df_view.columns else None
    )

# ── <target> probability distribution by outcome ─────────────────────────────
_score_col = f"prob_{target_class}"
if _score_col in df_view.columns and s1_truth is not None:
    st.markdown(f"### {target_class} probability distribution by outcome")
    st.caption(
        f"Binary view ({target_class} vs not-{target_class}) at threshold = "
        f"**{threshold:.2f}** (dashed line). Bars are stacked by outcome "
        "(TP / FP / FN / TN) on a log Y axis."
    )
    score = _series(df_view, _score_col).astype(float).clip(0.0, 1.0)
    mask_hist = score.notna() & s1_truth.notna()
    if int(mask_hist.sum()) == 0:
        st.info(f"No rows with both {_score_col} and the chosen reference.")
    else:
        s_arr = np.asarray(score[mask_hist].to_numpy())
        ref_pos = np.asarray(
            s1_truth[mask_hist].astype(str).eq(target_class).to_numpy()
        )
        pred_pos = s_arr >= threshold
        outcome = np.where(
            pred_pos & ref_pos,
            "TP",
            np.where(
                pred_pos & ~ref_pos, "FP", np.where(~pred_pos & ref_pos, "FN", "TN")
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
        outcome_colors = {
            "TP": "#2ca02c",
            "FP": "#d62728",
            "FN": "#ff7f0e",
            "TN": "#1f77b4",
        }
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
                    alt.Tooltip(
                        "bin_start:Q",
                        title=f"{_score_col} bin start",
                        format=".3f",
                    ),
                    alt.Tooltip("count:Q", title="Rows", format=","),
                ],
            )
        )
        thr_rule = (
            alt.Chart(pd.DataFrame({"thr": [threshold]}))
            .mark_rule(strokeDash=[4, 4], color="black")
            .encode(x=alt.X("thr:Q", scale=alt.Scale(domain=[0.0, 1.0])))
        )
        st.altair_chart(
            (hist + thr_rule).properties(height=320), use_container_width=True
        )

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
_macro_opt_thr = _optimal_ecotype_macro_fb_threshold(
    df_view, pred_3c_thr, s2_truth, classes_ec, beta
)
if _macro_opt_thr is not None:
    st.caption(
        f"💡 Threshold maximising **macro {beta_label}** across the 5 ecotypes "
        f"({objective.lower()}) vs the current reference: **{_macro_opt_thr:.2f}**. "
        f"Pipeline default: **{_default_abst:.2f}**."
    )
else:
    st.caption(f"Pipeline default: **{_default_abst:.2f}**.")

pred_ec_thr = _threshold_pred_ecotype(df_view, pred_3c_thr, unassigned_thr, classes_ec)
if pred_ec_thr is None:
    pred_ec_thr = (
        _series(df_view, "_pred_ecotype")
        if "_pred_ecotype" in df_view.columns
        else None
    )

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
