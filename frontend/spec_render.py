from pathlib import Path
from typing import Optional, Tuple

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from matplotlib.figure import Figure

SAMPLE_RATE = 24000
N_MELS = 224
WINDOW_SEC = 2.0
EXPANDED_SEC = 10.0
TOP_DB = 80.0
AC_PCTL_LO = 5
AC_PCTL_HI = 98
NOISE_FLOOR_AXIS = 1
AUDIO_ROOT = Path(
    "/home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/all_audios"
)


def hz_to_mel(f: float) -> float:
    return 2595.0 * np.log10(1.0 + f / 700.0)


def _process_spec(
    spec: np.ndarray, plot_gain: float, noise_reduction: bool
) -> np.ndarray:
    s = spec.astype(np.float32, copy=True)
    if plot_gain != 1.0:
        s = s + 10.0 * np.log10(plot_gain)
    if noise_reduction:
        nf = np.median(s, axis=NOISE_FLOOR_AXIS, keepdims=True)
        s = np.maximum(s - nf, s.min())
    return s


def _v_range(spec: np.ndarray, auto_contrast: bool) -> Tuple[float, float]:
    if auto_contrast:
        vmin, vmax = np.percentile(spec, [AC_PCTL_LO, AC_PCTL_HI])
        if vmax - vmin < 20:
            vmax = vmin + 20
    else:
        vmax = float(spec.max())
        vmin = vmax - TOP_DB
    return float(vmin), float(vmax)


def _to_linear_freq(spec: np.ndarray, sample_rate: int) -> np.ndarray:
    n_mels = spec.shape[0]
    mel_freqs = librosa.mel_frequencies(n_mels=n_mels, fmax=sample_rate / 2)
    linear_freqs = np.linspace(0, sample_rate / 2, n_mels)
    out = np.empty_like(spec)
    for t in range(spec.shape[1]):
        out[:, t] = np.interp(linear_freqs, mel_freqs, spec[:, t])
    return out


def _draw_axes(
    ax,
    n_frames: int,
    t_total: float,
    sample_rate: int,
    n_mels: int,
    linear_scale: bool,
) -> None:
    n_ticks = 5
    x_ticks = [t_total * i / (n_ticks - 1) for i in range(n_ticks)]
    ax.set_xticks([t / t_total * n_frames if t_total > 0 else 0 for t in x_ticks])
    ax.set_xticklabels([f"{t:.1f}" for t in x_ticks])
    ax.set_xlabel("Time (s)")

    if linear_scale:
        nyq_khz = sample_rate / 2 / 1000.0
        khz_ticks = np.arange(0, nyq_khz + 0.1, 2.0)
        ax.set_yticks([f * 1000.0 / (sample_rate / 2) * n_mels for f in khz_ticks])
        ax.set_yticklabels([f"{int(k)}k" if k >= 1 else "0" for k in khz_ticks])
        ax.set_ylabel("Frequency (kHz, linear)")
    else:
        max_mel = hz_to_mel(sample_rate / 2)
        hz_ticks = [500, 2000, 4000, 8000]
        ax.set_yticks([hz_to_mel(f) / max_mel * n_mels for f in hz_ticks])
        ax.set_yticklabels([f"{f}" if f < 1000 else f"{f // 1000}k" for f in hz_ticks])
        ax.set_ylabel("Frequency (Hz, mel-scale)")


def render_spectrogram(
    npy_path: str,
    auto_contrast: bool = False,
    noise_reduction: bool = False,
    linear_scale: bool = False,
    plot_gain: float = 1.0,
) -> Figure:
    spec = np.load(npy_path)
    return _render_common(
        spec,
        WINDOW_SEC,
        SAMPLE_RATE,
        auto_contrast=auto_contrast,
        noise_reduction=noise_reduction,
        linear_scale=linear_scale,
        plot_gain=plot_gain,
        red_lines=None,
    )


@st.cache_data(show_spinner=False, max_entries=32)
def _load_expanded_data(
    audio_basename: str, start_s: float, end_s: float, target_sec: float
) -> Optional[dict]:
    p = AUDIO_ROOT / f"{audio_basename}.wav"
    if not p.exists():
        return None
    info = sf.info(str(p))
    sr = info.samplerate
    seg_dur = end_s - start_s
    extra = (target_sec - seg_dur) / 2.0
    load_start = max(0, int((start_s - extra) * sr))
    load_end = min(info.frames, int((end_s + extra) * sr))
    audio, _ = sf.read(str(p), start=load_start, stop=load_end, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=256, fmax=sr / 2
    )
    S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    return {
        "spec": S_db,
        "t_window_start": (start_s * sr - load_start) / sr,
        "t_window_end": (end_s * sr - load_start) / sr,
        "t_total": (load_end - load_start) / sr,
        "sr": sr,
    }


def render_expanded_spectrogram(
    audio_basename: str,
    start_s: float,
    end_s: float,
    target_sec: float = EXPANDED_SEC,
    auto_contrast: bool = False,
    noise_reduction: bool = False,
    linear_scale: bool = False,
    plot_gain: float = 1.0,
) -> Optional[Figure]:
    data = _load_expanded_data(audio_basename, start_s, end_s, target_sec)
    if data is None:
        return None
    return _render_common(
        data["spec"],
        data["t_total"],
        data["sr"],
        auto_contrast=auto_contrast,
        noise_reduction=noise_reduction,
        linear_scale=linear_scale,
        plot_gain=plot_gain,
        red_lines=(data["t_window_start"], data["t_window_end"]),
    )


def _render_common(
    spec: np.ndarray,
    t_total: float,
    sample_rate: int,
    auto_contrast: bool,
    noise_reduction: bool,
    linear_scale: bool,
    plot_gain: float,
    red_lines: Optional[Tuple[float, float]],
) -> Figure:
    s = _process_spec(spec, plot_gain, noise_reduction)
    if linear_scale:
        s = _to_linear_freq(s, sample_rate)
    vmin, vmax = _v_range(s, auto_contrast)

    fig = Figure(figsize=(8, 4), dpi=110)
    ax = fig.add_subplot(111)
    im = ax.imshow(s, aspect="auto", origin="lower", cmap="magma", vmin=vmin, vmax=vmax)
    n_frames = s.shape[1]
    _draw_axes(ax, n_frames, t_total, sample_rate, s.shape[0], linear_scale)

    if red_lines is not None and t_total > 0:
        for t in red_lines:
            x = t / t_total * n_frames
            ax.axvline(x=x, color="red", linewidth=1.0)

    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    return fig
