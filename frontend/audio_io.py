"""Audio slice loader, EAR.LOG datetime lookup, and audio processing for the review UI."""

import os
import re
import numpy as np

try:
    import soundfile as sf
    _SF_AVAILABLE = True
except ImportError:
    _SF_AVAILABLE = False

try:
    import librosa
    _LIBROSA_AVAILABLE = True
except ImportError:
    _LIBROSA_AVAILABLE = False

try:
    from scipy import signal as sp_signal
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

# Root directory where Tuxedni .wav files live
WAV_ROOTS = [
    "/home/v-manoloc/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/all_audios",
    "/home/v-manoloc/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/ID118/wav",
    "/home/v-manoloc/shared/v-druizlopez/NOAA_Whales/DataInput_New/Tuxedni_channel_CI/ID120/wav",
]

HIGHPASS_CUTOFF = 300   # Hz
N_MELS = 128
TOP_DB = 80
EXPANDED_SEC = 10.0


def _find_wav(audio_basename: str) -> str | None:
    """Resolve audio basename (e.g. '120_00000008.e') to a .wav path."""
    candidates = [audio_basename, audio_basename + ".wav"]
    for root in WAV_ROOTS:
        for name in candidates:
            p = os.path.join(root, name)
            if os.path.isfile(p):
                return p
    return None


def load_audio_slice(audio_basename: str, start_s: float, end_s: float):
    """Return (samples_ndarray, sample_rate) or (None, None) if unavailable."""
    if not _SF_AVAILABLE:
        return None, None
    wav_path = _find_wav(audio_basename)
    if wav_path is None:
        return None, None
    try:
        info = sf.info(wav_path)
        sr = info.samplerate
        data, sr = sf.read(wav_path, start=int(start_s * sr), stop=int(end_s * sr),
                           always_2d=False, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        return data, sr
    except Exception:
        return None, None


def _apply_highpass(data: np.ndarray, sr: int) -> np.ndarray:
    """Apply a 2nd-order Butterworth high-pass filter at HIGHPASS_CUTOFF Hz (gentle rolloff)."""
    if not _SCIPY_AVAILABLE:
        return data
    b, a = sp_signal.butter(2, HIGHPASS_CUTOFF / (sr / 2), btype="high")
    return sp_signal.filtfilt(b, a, data).astype(np.float32)


def compute_expanded_spectrogram(
    audio_basename: str, start_s: float, end_s: float, highpass: bool = False
) -> dict | None:
    """Compute a ~10 s mel-spectrogram centred on the 2 s segment.

    Returns a dict with keys: spec, t_start, t_end, t_total, audio, sr
    or None if unavailable.
    """
    if not (_SF_AVAILABLE and _LIBROSA_AVAILABLE):
        return None
    wav_path = _find_wav(audio_basename)
    if wav_path is None:
        return None
    try:
        info = sf.info(wav_path)
        sr = info.samplerate
        seg_dur = end_s - start_s
        pad = (EXPANDED_SEC - seg_dur) / 2
        load_start = max(0, int((start_s - pad) * sr))
        load_end = min(info.frames, int((end_s + pad) * sr))
        data, _ = sf.read(wav_path, start=load_start, stop=load_end,
                          always_2d=False, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)

        if highpass:
            data = _apply_highpass(data, sr)

        S = librosa.feature.melspectrogram(
            y=data, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512, fmax=sr / 2)
        S_db = librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)

        t_start = start_s - load_start / sr
        t_end = end_s - load_start / sr
        t_total = (load_end - load_start) / sr

        return {"spec": S_db, "t_start": t_start, "t_end": t_end,
                "t_total": t_total, "audio": data, "sr": sr}
    except Exception:
        return None


def compute_2s_spectrogram(
    audio_basename: str, start_s: float, end_s: float, highpass: bool = False
) -> np.ndarray | None:
    """Compute mel-spectrogram for the 2 s segment with optional HPF.

    Returns a (n_mels, T) dB array, or None if unavailable.
    Used when the pre-computed .npy would not reflect the HPF.
    """
    if not (_SF_AVAILABLE and _LIBROSA_AVAILABLE):
        return None
    data, sr = load_audio_slice(audio_basename, start_s, end_s)
    if data is None:
        return None
    try:
        if highpass:
            data = _apply_highpass(data, sr)
        S = librosa.feature.melspectrogram(
            y=data, sr=sr, n_mels=N_MELS, n_fft=2048, hop_length=512, fmax=sr / 2)
        return librosa.power_to_db(S, ref=np.max, top_db=TOP_DB)
    except Exception:
        return None


def apply_audio_processing(
    data: np.ndarray,
    sr: int,
    playback_gain: float = 1.0,
    highpass: bool = False,
    noise_reduction: bool = False,
) -> tuple[np.ndarray, int]:
    """Apply gain, optional high-pass filter, noise reduction, and soft-clip.
    Returns (processed, out_sr).
    """
    audio = data.astype(np.float32)

    # Resample to 44100 for browser compatibility
    target_sr = 44100
    if _LIBROSA_AVAILABLE and sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    else:
        target_sr = sr

    # High-pass filter at HIGHPASS_CUTOFF Hz
    if highpass and _SCIPY_AVAILABLE:
        b, a = sp_signal.butter(2, HIGHPASS_CUTOFF / (target_sr / 2), btype="high")
        audio = sp_signal.filtfilt(b, a, audio)

    # Spectral subtraction noise reduction
    if noise_reduction and _SCIPY_AVAILABLE:
        nperseg = 512
        f, t, Zxx = sp_signal.stft(audio, fs=target_sr, nperseg=nperseg)
        magnitude = np.abs(Zxx)
        phase = np.angle(Zxx)
        # Subtract only 50% of the estimated noise floor and keep at least
        # 40% of the original magnitude — gentler reduction, fewer artifacts.
        noise_floor = np.median(magnitude, axis=1, keepdims=True)
        magnitude_clean = np.maximum(magnitude - 0.5 * noise_floor, 0.4 * magnitude)
        _, audio = sp_signal.istft(magnitude_clean * np.exp(1j * phase),
                                   fs=target_sr, nperseg=nperseg)
        audio = audio.astype(np.float32)

    # Gain
    audio = audio * playback_gain

    # Soft clip
    threshold = 0.95
    mask = np.abs(audio) > threshold
    audio[mask] = np.sign(audio[mask]) * (
        threshold + (1.0 - threshold) * np.tanh(
            (np.abs(audio[mask]) - threshold) / (1.0 - threshold)))

    return np.clip(audio, -1.0, 1.0).astype(np.float32), target_sr


# ── EAR.LOG datetime lookup ──────────────────────────────────────────────────

EAR_LOG_PATH = "/home/v-manoloc/shared/v-manoloc/EAR.LOG"
_LOG_PATTERN = re.compile(
    r"^(\d{4}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})\s+Recording to\s+(\d+)\.",
    re.IGNORECASE,
)


def parse_ear_log(log_path: str = EAR_LOG_PATH) -> dict:
    """Parse EAR.LOG and return {file_number_str: datetime_str}."""
    mapping = {}
    try:
        with open(log_path, "r", errors="replace") as f:
            for line in f:
                m = _LOG_PATTERN.match(line.strip())
                if m:
                    mapping[m.group(2)] = m.group(1)
    except FileNotFoundError:
        pass
    return mapping


def get_recording_datetime(audio_basename: str, log_mapping: dict) -> str | None:
    """Return recording start datetime for '120_00000008.e' → looks up '00000008'."""
    stem = audio_basename.split("_", 1)[-1]
    file_num = stem.split(".")[0]
    return log_mapping.get(file_num)
