import os
import math
import random
import numpy as np
import soundfile as sf
from pathlib import Path


EPS = 1e-12

import librosa
from pathlib import Path

def collect_esc50_noise(
    esc50_root: str,
    target_sr: int,
    max_files: int = None,
):
    """
    Collect ESC-50 noise wavs and resample to target_sr.
    """
    esc_root = Path(esc50_root)
    wavs = sorted(esc_root.rglob("*.wav"))

    if max_files is not None:
        wavs = wavs[:max_files]

    noise_pool = []
    for wp in wavs:
        noise, sr = librosa.load(wp, sr=None, mono=True)
        if sr != target_sr:
            noise = librosa.resample(noise, orig_sr=sr, target_sr=target_sr)
        noise_pool.append(noise)

    print(f"[INFO] Loaded {len(noise_pool)} ESC-50 noise files")
    return noise_pool

def signal_power(x: np.ndarray) -> float:
    """Mean square power."""
    return float(np.mean(x.astype(np.float64) ** 2) + EPS)


def normalize_audio(x: np.ndarray) -> np.ndarray:
    """Prevent clipping after noise addition."""
    max_val = np.max(np.abs(x))
    if max_val > 1.0:
        x = x / (max_val + EPS)
    return x


def add_awgn(signal: np.ndarray, snr_db: float, rng: np.random.Generator):
    """
    Add white Gaussian noise to reach target SNR (dB).
    """
    ps = signal_power(signal)
    pn = ps / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, math.sqrt(pn), size=signal.shape)
    return normalize_audio(signal + noise)


def add_real_noise(signal: np.ndarray,
                   noise: np.ndarray,
                   snr_db: float,
                   rng: np.random.Generator):
    """
    Add real noise waveform with target SNR.
    """
    L = len(signal)

    if len(noise) < L:
        repeat = int(np.ceil(L / len(noise)))
        noise = np.tile(noise, repeat)[:L]
    else:
        start = rng.integers(0, len(noise) - L + 1)
        noise = noise[start:start + L]

    ps = signal_power(signal)
    pn_target = ps / (10.0 ** (snr_db / 10.0))
    pn = signal_power(noise)

    scale = math.sqrt(pn_target / pn)
    noise = noise * scale

    return normalize_audio(signal + noise)


def make_noisy_testset(
    clean_root: str,
    noisy_root: str,
    snr_db: float,
    noise_pool: list = None,   # 注意：不再是文件路径
    seed: int = 42,
    overwrite: bool = True,
):
    rng = np.random.default_rng(seed)
    clean_root = Path(clean_root)
    noisy_root = Path(noisy_root)
    noisy_root.mkdir(parents=True, exist_ok=True)

    for wav_path in clean_root.rglob("*.wav"):
        rel_path = wav_path.relative_to(clean_root)
        out_path = noisy_root / rel_path
        out_path.parent.mkdir(parents=True, exist_ok=True)

        if out_path.exists() and not overwrite:
            continue

        signal, fs = sf.read(wav_path, dtype="float32")
        if signal.ndim > 1:
            signal = np.mean(signal, axis=1)

        if noise_pool is None:
            noisy = add_awgn(signal, snr_db, rng)
        else:
            noise = rng.choice(noise_pool)
            noisy = add_real_noise(signal, noise, snr_db, rng)

        sf.write(out_path, noisy, fs)

    print(f"[OK] Noisy test set generated: {noisy_root} (SNR={snr_db} dB)")

