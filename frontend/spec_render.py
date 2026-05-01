"""Spectrogram rendering helper — does NOT import plot_spectrograms.py
to avoid the forced matplotlib.use('Agg') side-effect at module level.
Only the 5 lines we need are copied here.
"""

import numpy as np
import matplotlib.figure

SAMPLE_RATE = 24000
NYQUIST = SAMPLE_RATE / 2
TOP_DB = 80
CATEGORY_MAP = {0: "Background", 1: "Humpback", 2: "Orca", 3: "Beluga"}
LABEL_COLORS = {0: "white", 1: "cyan", 2: "yellow", 3: "lime"}
TITLE_COLORS = {0: "#666666", 1: "#b34700", 2: "#5b0080", 3: "#1a6b1a"}

_HZ_TICKS = [500, 1000, 2000, 4000, 6000, 8000, 10000, int(NYQUIST)]


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def mel_to_hz(m: float) -> float:
    return 700.0 * (10.0 ** (m / 2595.0) - 1.0)


def _remap_path(path: str) -> str:
    return path.replace(
        "/home/v-druizlopez/shared/",
        "/home/v-manoloc/shared/v-druizlopez/../",
    )


def _to_linear(spec: np.ndarray) -> np.ndarray:
    """Resample a mel spectrogram (n_mels × T) to a linear-Hz grid of the same height."""
    n_mels = spec.shape[0]
    mel_max = hz_to_mel(NYQUIST)
    mel_rows = np.linspace(0, mel_max, n_mels)
    hz_rows = mel_to_hz(mel_rows)
    hz_linear = np.linspace(hz_rows[0], hz_rows[-1], n_mels)
    linear = np.empty_like(spec)
    for t in range(spec.shape[1]):
        linear[:, t] = np.interp(hz_linear, hz_rows, spec[:, t])
    return linear


def _apply_processing(
    spec: np.ndarray,
    plot_gain: float = 1.0,
    noise_reduction: bool = False,
    auto_contrast: bool = False,
    hpf_cutoff_hz: float | None = None,   # when set, exclude HPF-zeroed bins from percentile
):
    """Apply plot-gain, noise-reduction, and compute vmin/vmax for imshow."""
    S = spec.copy().astype(np.float32)
    n_mels = S.shape[0]

    # Plot gain in dB domain
    if plot_gain != 1.0:
        S = S + 10.0 * np.log10(max(plot_gain, 1e-6))

    # Noise reduction: subtract per-frequency median
    if noise_reduction:
        noise_floor = np.median(S, axis=1, keepdims=True)
        S = np.maximum(S - noise_floor, S.min())

    # Contrast / colour range
    if auto_contrast:
        # When HPF is active, low-frequency bins are near-zero and would
        # drag the 5th-percentile down, washing out the rest of the image.
        # Restrict percentile calculation to bins above the HPF cutoff.
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
            vmin, vmax = S.min(), S.max()
        if vmax - vmin < 20:
            vmax = vmin + 20
    else:
        vmax = S.max()
        vmin = vmax - TOP_DB

    # Safety clamps
    if vmin > S.min() - 2.0:
        vmin = S.min() - 2.0
    if vmax - vmin < 20.0:
        vmin = vmax - 20.0
    if np.isclose(vmin, vmax):
        vmin = vmax - 1.0

    return S, float(vmin), float(vmax)


def _tick_label(hz: int) -> str:
    return f"{hz // 1000}k" if hz >= 1000 else str(hz)


def render_spectrogram(
    npy_path: str,
    pred_label: int | None = None,
    scale: str = "mel",
    auto_contrast: bool = False,
    noise_reduction: bool = False,
    plot_gain: float = 1.0,
    highpass: bool = False,
    expanded_spec: np.ndarray | None = None,
    t_markers: tuple | None = None,
    t_total: float | None = None,
) -> matplotlib.figure.Figure:
    """Render a spectrogram figure with all processing applied."""
    from audio_io import HIGHPASS_CUTOFF  # avoid circular import at module level

    if expanded_spec is not None:
        spec = expanded_spec.astype(np.float32)
    else:
        resolved = _remap_path(npy_path)
        spec = np.load(resolved).astype(np.float32)

    n_mels = spec.shape[0]
    mel_max = hz_to_mel(NYQUIST)

    # Processing pipeline
    S, vmin, vmax = _apply_processing(
        spec, plot_gain, noise_reduction, auto_contrast,
        hpf_cutoff_hz=HIGHPASS_CUTOFF if highpass else None,
    )

    # Frequency axis resampling
    if scale == "linear":
        display = _to_linear(S)
        def _hz_to_row(hz):
            hz_min = mel_to_hz(0)
            return (hz - hz_min) / (NYQUIST - hz_min) * (n_mels - 1)
        tick_positions = [_hz_to_row(hz) for hz in _HZ_TICKS]
        freq_label = "Frequency — Linear Hz"
    else:
        display = S
        tick_positions = [hz_to_mel(hz) / mel_max * (n_mels - 1) for hz in _HZ_TICKS]
        freq_label = "Frequency — Mel (Hz)"

    # Time axis
    n_frames = display.shape[1]
    duration = t_total if t_total is not None else 2.0
    extent = [0, duration, 0, n_mels]

    fig = matplotlib.figure.Figure(figsize=(8, 3), dpi=120)
    ax = fig.add_subplot(1, 1, 1)

    ax.imshow(display, aspect="auto", origin="lower", cmap="magma",
              vmin=vmin, vmax=vmax,
              extent=extent, interpolation="nearest")

    # Hz ticks
    valid = [(pos, hz) for pos, hz in zip(tick_positions, _HZ_TICKS) if 0 <= pos <= n_mels]
    if valid:
        positions, labels = zip(*valid)
        ax.set_yticks(list(positions))
        ax.set_yticklabels([_tick_label(hz) for hz in labels], fontsize=7)

    ax.set_xlabel("Time (s)" if t_total else "Time frames", fontsize=8)
    ax.set_ylabel(freq_label, fontsize=8)

    # Red delimiter lines for expanded view
    if t_markers is not None:
        ax.axvline(x=t_markers[0], color="red", linewidth=1.2, linestyle="--")
        ax.axvline(x=t_markers[1], color="red", linewidth=1.2, linestyle="--")

    if pred_label is not None:
        title = f"pred: {CATEGORY_MAP.get(pred_label, str(pred_label))}"
        title_color = TITLE_COLORS.get(pred_label, "black")
    else:
        title = ""
        title_color = "black"
    ax.set_title(title, fontsize=9, color=title_color)

    fig.tight_layout()
    return fig
