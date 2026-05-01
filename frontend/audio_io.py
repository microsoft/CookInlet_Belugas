import io
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf
import streamlit as st
from scipy import signal as sp_signal

AUDIO_ROOT = Path(
    "/home/v-druizlopez/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/all_audios"
)
HIGHPASS_HZ = 500.0


def resolve_audio_path(audio_basename: str) -> Optional[Path]:
    p = AUDIO_ROOT / f"{audio_basename}.wav"
    return p if p.exists() else None


def _apply_processing(
    audio: np.ndarray, sr: int, gain: float, highpass: bool
) -> np.ndarray:
    if highpass:
        b, a = sp_signal.butter(4, HIGHPASS_HZ / (sr / 2), btype="high")
        audio = sp_signal.filtfilt(b, a, audio)
    audio = audio * gain
    threshold = 0.95
    mask = np.abs(audio) > threshold
    if np.any(mask):
        audio[mask] = np.sign(audio[mask]) * (
            threshold
            + (1.0 - threshold)
            * np.tanh((np.abs(audio[mask]) - threshold) / (1.0 - threshold))
        )
    return np.clip(audio, -1.0, 1.0).astype(np.float32)


@st.cache_data(show_spinner=False, max_entries=64)
def load_audio_slice_wav(
    audio_basename: str,
    start_s: float,
    end_s: float,
    expanded: bool = False,
    target_sec: float = 10.0,
    gain: float = 1.0,
    highpass: bool = False,
) -> Optional[bytes]:
    p = resolve_audio_path(audio_basename)
    if p is None:
        return None
    info = sf.info(str(p))
    sr = info.samplerate

    if expanded:
        seg_dur = end_s - start_s
        extra = (target_sec - seg_dur) / 2.0
        load_start = max(0, int((start_s - extra) * sr))
        load_end = min(info.frames, int((end_s + extra) * sr))
    else:
        load_start = int(start_s * sr)
        load_end = int(end_s * sr)

    audio, _ = sf.read(str(p), start=load_start, stop=load_end, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    audio = _apply_processing(audio, sr, gain, highpass)

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()
