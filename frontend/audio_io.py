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
            str(p), start=load_start, stop=load_end,
            always_2d=False, dtype="float32",
        )
        if data.ndim > 1:
            data = data.mean(axis=1)
        if highpass:
            data = _highpass(data, sr)

        S = librosa.feature.melspectrogram(
            y=data, sr=sr, n_mels=config.N_MELS,
            n_fft=2048, hop_length=512, fmax=sr / 2,
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
        if highpass:
            data = _highpass(data, sr)
        S = librosa.feature.melspectrogram(
            y=data, sr=sr, n_mels=config.N_MELS,
            n_fft=2048, hop_length=512, fmax=sr / 2,
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
    reduction → soft-clip. Resamples to `PLAYBACK_SAMPLE_RATE` for browser
    compatibility. Returns (processed, out_sr).
    """
    audio = data.astype(np.float32)

    target_sr = config.PLAYBACK_SAMPLE_RATE
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)

    if highpass:
        b, a = sp_signal.butter(2, config.HIGHPASS_CUTOFF_HZ / (target_sr / 2), btype="high")  # type: ignore[misc]
        audio = sp_signal.filtfilt(b, a, audio).astype(np.float32)

    if noise_reduction:
        nperseg = 512
        _, _, Zxx = sp_signal.stft(audio, fs=target_sr, nperseg=nperseg)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        noise_floor = np.median(magnitude, axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - 0.5 * noise_floor, 0.4 * magnitude)
        _, audio = sp_signal.istft(
            magnitude_clean * np.exp(1j * phase), fs=target_sr, nperseg=nperseg
        )
        audio = audio.astype(np.float32)

    audio = audio * playback_gain

    threshold = 0.95
    mask = np.abs(audio) > threshold
    if np.any(mask):
        audio[mask] = np.sign(audio[mask]) * (
            threshold + (1.0 - threshold) * np.tanh(
                (np.abs(audio[mask]) - threshold) / (1.0 - threshold)
            )
        )

    return np.clip(audio, -1.0, 1.0).astype(np.float32), target_sr


def encode_wav(data: np.ndarray, sr: int) -> bytes:
    """Encode a numpy float32 array as in-memory WAV bytes for st.audio."""
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="FLOAT")
    return buf.getvalue()
