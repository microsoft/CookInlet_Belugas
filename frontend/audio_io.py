"""Audio loading and playback DSP for the review UI.

Reads `.wav` files from `config.AUDIO_ROOT`, returns either raw samples or
ready-to-play WAV bytes. Optional gain, high-pass filter, and spectral-
subtraction noise reduction are applied at playback time only — the audio
on disk is never modified.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import streamlit as st
from scipy import signal as sp_signal

import config


def _resolve_wav(audio_basename: str) -> Optional[Path]:
    """Resolve a CSV `audio` value (e.g. '120_00000008.e') to a .wav path."""
    if config.AUDIO_ROOT is None:
        return None
    candidates = [audio_basename, f"{audio_basename}.wav"]
    for name in candidates:
        p = config.AUDIO_ROOT / name
        if p.is_file():
            return p
    return None


@st.cache_data(show_spinner=False, max_entries=128)
def load_audio_slice(
    audio_basename: str, start_s: float, end_s: float
) -> tuple[Optional[np.ndarray], Optional[int]]:
    """Return (samples, sample_rate) for the requested window, or (None, None)."""
    p = _resolve_wav(audio_basename)
    if p is None:
        return None, None
    try:
        info = sf.info(str(p))
        sr = info.samplerate
        data, sr = sf.read(
            str(p),
            start=int(start_s * sr),
            stop=int(end_s * sr),
            always_2d=False,
            dtype="float32",
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        return None, None


def _highpass(data: np.ndarray, sr: int) -> np.ndarray:
    b, a = sp_signal.butter(2, config.HIGHPASS_CUTOFF_HZ / (sr / 2), btype="high")  # type: ignore[misc]
    return sp_signal.filtfilt(b, a, data).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=64)
def compute_expanded_spectrogram(
    audio_basename: str,
    start_s: float,
    end_s: float,
    highpass: bool = False,
) -> Optional[dict]:
    """Compute a mel-spectrogram on a `EXPANDED_VIEW_SEC`-second window centred
    on the segment. Returns dict with keys: spec, t_start, t_end, t_total,
    audio, sr — or None if unavailable.
    """
    p = _resolve_wav(audio_basename)
    if p is None:
        return None
    try:
        info = sf.info(str(p))
        sr = info.samplerate
        seg_dur = end_s - start_s
        pad = max(0.0, (config.EXPANDED_VIEW_SEC - seg_dur) / 2.0)
        load_start = max(0, int((start_s - pad) * sr))
        load_end = min(info.frames, int((end_s + pad) * sr))
        data, _ = sf.read(
            str(p),
            start=load_start,
            stop=load_end,
            always_2d=False,
            dtype="float32",
        )
        if data.ndim > 1:
            data = data.mean(axis=1)

        # Resample to config.SAMPLE_RATE before STFT so n_fft/hop give consistent
        # time/freq resolution across files with different native sample rates;
        # otherwise high-SR files leave the bottom mel bins under-resolved.
        if sr != config.SAMPLE_RATE:
            spec_data = librosa.resample(data, orig_sr=sr, target_sr=config.SAMPLE_RATE)
        else:
            spec_data = data
        spec_sr = config.SAMPLE_RATE

        if highpass:
            spec_data = _highpass(spec_data, spec_sr)

        S = librosa.feature.melspectrogram(
            y=spec_data,
            sr=spec_sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            fmax=spec_sr / 2,
        )
        S_db = librosa.power_to_db(S, ref=np.max, top_db=config.TOP_DB)

        return {
            "spec": S_db,
            "t_start": start_s - load_start / sr,
            "t_end": end_s - load_start / sr,
            "t_total": (load_end - load_start) / sr,
            "audio": data,
            "sr": sr,
        }
    except Exception:
        return None


@st.cache_data(show_spinner=False, max_entries=64)
def compute_segment_spectrogram(
    audio_basename: str,
    start_s: float,
    end_s: float,
    highpass: bool = False,
) -> Optional[np.ndarray]:
    """Recompute the segment's mel-spectrogram (used when high-pass is on, as
    the pre-computed .npy on disk wouldn't reflect the filter)."""
    data, sr = load_audio_slice(audio_basename, start_s, end_s)
    if data is None or sr is None:
        return None
    try:
        if sr != config.SAMPLE_RATE:
            data = librosa.resample(data, orig_sr=sr, target_sr=config.SAMPLE_RATE)
            sr = config.SAMPLE_RATE
        if highpass:
            data = _highpass(data, sr)
        S = librosa.feature.melspectrogram(
            y=data,
            sr=sr,
            n_mels=config.N_MELS,
            n_fft=config.N_FFT,
            hop_length=config.HOP_LENGTH,
            fmax=sr / 2,
        )
        return librosa.power_to_db(S, ref=np.max, top_db=config.TOP_DB)
    except Exception:
        return None


def apply_audio_processing(
    data: np.ndarray,
    sr: int,
    playback_gain: float = 1.0,
    highpass: bool = False,
    noise_reduction: bool = False,
) -> tuple[np.ndarray, int]:
    """Apply gain → optional HPF → optional spectral-subtraction noise
    reduction → soft-clip. Keeps the source sample rate (no resampling — the
    browser plays WAV at any rate natively, and resampling 16 kHz FLAC up to
    44.1 kHz introduced audible artifacts). Only downsamples if the source
    SR is above 48 kHz, which would otherwise inflate the WAV size needlessly.
    Returns (processed, out_sr).
    """
    audio = data.astype(np.float32)

    out_sr = sr
    if sr > 48000:
        out_sr = 48000
        audio = librosa.resample(audio, orig_sr=sr, target_sr=out_sr)

    if highpass:
        b, a = sp_signal.butter(
            2, config.HIGHPASS_CUTOFF_HZ / (out_sr / 2), btype="high"
        )  # type: ignore[misc]
        audio = sp_signal.filtfilt(b, a, audio).astype(np.float32)

    if noise_reduction:
        nperseg = min(512, max(64, len(audio) // 4))
        _, _, Zxx = sp_signal.stft(audio, fs=out_sr, nperseg=nperseg)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        noise_floor = np.median(magnitude, axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - 0.5 * noise_floor, 0.4 * magnitude)
        _, audio = sp_signal.istft(
            magnitude_clean * np.exp(1j * phase), fs=out_sr, nperseg=nperseg
        )
        audio = audio.astype(np.float32)

    audio = audio * playback_gain

    threshold = 0.95
    mask = np.abs(audio) > threshold
    if np.any(mask):
        audio[mask] = np.sign(audio[mask]) * (
            threshold
            + (1.0 - threshold)
            * np.tanh((np.abs(audio[mask]) - threshold) / (1.0 - threshold))
        )

    return np.clip(audio, -1.0, 1.0).astype(np.float32), out_sr


def encode_wav(data: np.ndarray, sr: int) -> bytes:
    """Encode samples as in-memory PCM-16 WAV for st.audio.

    PCM-16 is the universally-supported WAV subtype; 32-bit float WAV (the
    previous default) is decoded inconsistently across browsers and can
    produce audible clicks and pops on playback.
    """
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@st.cache_data(show_spinner=False, max_entries=64)
def get_segment_wav_bytes(
    audio_basename: str,
    start_s: float,
    end_s: float,
    playback_gain: float,
    highpass: bool,
    noise_reduction: bool,
) -> Optional[bytes]:
    """Cached end-to-end: load segment → process → encode WAV bytes."""
    data, sr = load_audio_slice(audio_basename, start_s, end_s)
    if data is None or sr is None:
        return None
    processed, out_sr = apply_audio_processing(
        data, sr,
        playback_gain=playback_gain,
        highpass=highpass,
        noise_reduction=noise_reduction,
    )
    return encode_wav(processed, out_sr)


@st.cache_data(show_spinner=False, max_entries=32)
def get_expanded_wav_bytes(
    audio_basename: str,
    start_s: float,
    end_s: float,
    playback_gain: float,
    highpass: bool,
    noise_reduction: bool,
) -> Optional[bytes]:
    """Cached end-to-end: load 10-s window → process → encode WAV bytes."""
    exp = compute_expanded_spectrogram(audio_basename, start_s, end_s, highpass=False)
    if exp is None:
        return None
    processed, out_sr = apply_audio_processing(
        exp["audio"], exp["sr"],
        playback_gain=playback_gain,
        highpass=highpass,
        noise_reduction=noise_reduction,
    )
    return encode_wav(processed, out_sr)
