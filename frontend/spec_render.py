"""Spectrogram rendering for the review UI.

Renders a `.npy` mel-spectrogram (or an in-memory recompute) as a Matplotlib
Figure with mel-scale or linear-Hz frequency axis, optional auto-contrast,
noise reduction, gain shift, and a colored title indicating the predicted
class. Does NOT import the project's `data/plot_spectrograms.py` to avoid
its module-level `matplotlib.use("Agg")` side-effect.
"""

from __future__ import annotations

from typing import Optional, Sequence

import matplotlib.figure
import numpy as np

import config

NYQUIST = config.SAMPLE_RATE / 2

_HZ_TICKS: list[int] = [500, 1000, 2000, 4000, 6000, 8000, 10000, int(NYQUIST)]


def hz_to_mel(f):
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m):
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _to_linear(spec: np.ndarray) -> np.ndarray:
    """Resample a mel spectrogram to a linear-Hz grid of the same height."""
    n_mels = spec.shape[0]
    mel_max = hz_to_mel(NYQUIST)
    mel_rows = np.linspace(0, mel_max, n_mels)
    hz_rows = np.asarray(mel_to_hz(mel_rows))
    hz_linear = np.linspace(float(hz_rows[0]), float(hz_rows[-1]), n_mels)
    linear = np.empty_like(spec)
    for t in range(spec.shape[1]):
        linear[:, t] = np.interp(hz_linear, hz_rows, spec[:, t])
    return linear


def _apply_processing(
    spec: np.ndarray,
    spec_gain: float,
    noise_reduction: bool,
    auto_contrast: bool,
    hpf_cutoff_hz: Optional[float],
) -> tuple[np.ndarray, float, float]:
    """Per-bin noise floor, contrast window, and gain-driven dB shift.

    `spec_gain` shifts the colormap window in dB rather than the data, so
    higher gain → window shifts down → data appears brighter, independent
    of auto-contrast.
    """
    S = spec.copy().astype(np.float32)
    n_mels = S.shape[0]

    if noise_reduction:
        noise_floor = np.median(S, axis=1, keepdims=True)
        S = np.maximum(S - noise_floor, S.min())

    if auto_contrast:
        if hpf_cutoff_hz is not None and hpf_cutoff_hz > 0:
            mel_max = hz_to_mel(NYQUIST)
            first_bin = int(hz_to_mel(hpf_cutoff_hz) / mel_max * (n_mels - 1))
            first_bin = min(first_bin + 1, n_mels - 1)
            S_roi = S[first_bin:, :].flatten()
        else:
            S_roi = S.flatten()
        try:
            vmin, vmax = np.percentile(S_roi, [5, 98])
        except Exception:
            vmin, vmax = float(S.min()), float(S.max())
        if vmax - vmin < 20:
            vmax = vmin + 20
    else:
        vmax = float(S.max())
        vmin = vmax - config.TOP_DB

    if spec_gain != 1.0:
        gain_db = 10.0 * np.log10(max(spec_gain, 1e-6))
        vmin -= gain_db
        vmax -= gain_db

    if np.isclose(vmin, vmax):
        vmin = vmax - 1.0

    return S, float(vmin), float(vmax)


def _tick_label(hz: int) -> str:
    return f"{hz // 1000}k" if hz >= 1000 else str(hz)


def render_spectrogram(
    npy_path: str,
    pred_label: Optional[int] = None,
    scale: str = "mel",
    auto_contrast: bool = False,
    noise_reduction: bool = False,
    spec_gain: float = 1.0,
    highpass: bool = False,
    expanded_spec: Optional[np.ndarray] = None,
    t_markers: Optional[Sequence[float]] = None,
    t_total: Optional[float] = None,
    cmap: str = "magma",
) -> matplotlib.figure.Figure:
    """Render a spectrogram figure with all processing applied.

    If `expanded_spec` is provided it is rendered directly (useful when the
    caller has recomputed a longer or HPF-filtered spectrogram); otherwise
    the spectrogram is loaded from `npy_path`. `t_markers` draws vertical
    red lines at the given seconds (e.g. to delimit the segment within an
    expanded view). `t_total` is the duration of the rendered window.
    """
    if expanded_spec is not None:
        spec = expanded_spec.astype(np.float32)
    else:
        spec = np.load(npy_path).astype(np.float32)

    n_mels = spec.shape[0]
    mel_max = hz_to_mel(NYQUIST)

    S, vmin, vmax = _apply_processing(
        spec,
        spec_gain,
        noise_reduction,
        auto_contrast,
        hpf_cutoff_hz=config.HIGHPASS_CUTOFF_HZ if highpass else None,
    )

    if scale == "linear":
        display = _to_linear(S)
        def _hz_to_row(hz: float) -> float:
            return hz / NYQUIST * (n_mels - 1)
        tick_positions = [_hz_to_row(hz) for hz in _HZ_TICKS]
        freq_label = "Frequency — Linear Hz"
    else:
        display = S
        tick_positions = [hz_to_mel(hz) / mel_max * (n_mels - 1) for hz in _HZ_TICKS]
        freq_label = "Frequency — Mel (Hz)"

    duration = float(t_total) if t_total is not None else config.SEGMENT_VIEW_SEC
    extent = (0.0, duration, 0.0, float(n_mels))

    fig = matplotlib.figure.Figure(figsize=(8, 3), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(
        display, aspect="auto", origin="lower", cmap=cmap,
        vmin=vmin, vmax=vmax, extent=extent, interpolation="nearest",
    )

    valid = [(pos, hz) for pos, hz in zip(tick_positions, _HZ_TICKS) if 0 <= pos <= n_mels]
    if valid:
        positions, labels = zip(*valid)
        ax.set_yticks(list(positions))
        ax.set_yticklabels([_tick_label(hz) for hz in labels], fontsize=7)

    ax.set_xlabel("Time (s)" if t_total else "Time frames", fontsize=8)
    ax.set_ylabel(freq_label, fontsize=8)

    if t_markers is not None:
        for t in t_markers:
            ax.axvline(x=t, color="red", linewidth=1.2, linestyle="--")

    if pred_label is not None:
        title = f"pred: {config.PRED_LABELS.get(pred_label, str(pred_label))}"
        title_color = config.PRED_TITLE_COLORS.get(pred_label, "black")
        ax.set_title(title, fontsize=9, color=title_color)

    fig.tight_layout()
    return fig
