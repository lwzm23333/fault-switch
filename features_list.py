import os
import numpy as np
import pandas as pd
import librosa
from scipy.signal import hilbert, find_peaks
from scipy.stats import kurtosis
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from typing import List, Tuple, Optional

# --------------------------
# 特征函数（修正版，带异常保护）
# --------------------------

def calculate_energy(signal: np.ndarray) -> float:
    return float(np.sum(signal ** 2))


def calculate_zero_crossing_rate(signal: np.ndarray) -> float:
    if len(signal) < 2:
        return 0.0
    zc = np.sum(signal[:-1] * signal[1:] < 0)
    return float(zc / (len(signal) - 1))


def calculate_short_time_avg_magnitude(signal: np.ndarray) -> float:
    return float(np.mean(np.abs(signal))) if len(signal) > 0 else 0.0


def calculate_rms(signal: np.ndarray) -> float:
    return float(np.sqrt(np.mean(signal ** 2))) if len(signal) > 0 else 0.0


def calculate_peak_to_peak(signal: np.ndarray) -> float:
    return float(np.ptp(signal)) if len(signal) > 0 else 0.0



# ---- 频域 ----
def calculate_spectral_centroid(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        c = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft)
        return float(np.nan_to_num(np.mean(c)))
    except Exception:
        return 0.0



def calculate_spectral_roll_off(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        r = librosa.feature.spectral_rolloff(y=signal, sr=sr, n_fft=n_fft)
        return float(np.nan_to_num(np.mean(r)))
    except Exception:
        return 0.0


def calculate_spectral_flatness(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        f = librosa.feature.spectral_flatness(y=signal, n_fft=n_fft)
        return float(np.nan_to_num(np.mean(f)))
    except Exception:
        return 0.0


def calculate_spectral_contrast(signal: np.ndarray, sr: int, n_fft: int = 2048) -> np.ndarray:
    try:
        contrast = librosa.feature.spectral_contrast(y=signal, sr=sr, n_fft=n_fft)
        mean_v = np.nan_to_num(np.mean(contrast, axis=1))
        std_v = np.nan_to_num(np.std(contrast, axis=1))
        return np.concatenate([mean_v, std_v])
    except Exception:
        # 返回 14 个零（7 bands mean + 7 bands std）
        return np.zeros(14, dtype=float)


# ---- MFCCs + deltas ----
def calculate_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        return np.concatenate(np.mean(mfccs, axis=1))
    except Exception:
        return np.zeros(n_mfcc, dtype=float)

def calculate_delta_delta_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        d2 = librosa.feature.delta(mfccs, order=2)
        return np.nan_to_num(np.mean(d2, axis=1))
    except Exception:
        return np.zeros(n_mfcc, dtype=float)


# ---- 时频 ----
def calculate_stft_features(signal: np.ndarray, sr: int, n_fft: int = 512) -> float:
    try:
        S = np.abs(librosa.stft(signal, n_fft=n_fft))
        return float(np.nan_to_num(np.mean(S)))
    except Exception:
        return 0.0



# ---- Pitch (pyin) ----
def calculate_pitch_features(signal: np.ndarray, sr: int, fmin: float = 50.0, fmax: float = 1000.0) -> np.ndarray:
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(signal, fmin=fmin, fmax=fmax, sr=sr)
        f0_clean = np.nan_to_num(f0)  # NaN -> 0
        mean_f0 = float(np.mean(f0_clean)) if f0_clean.size > 0 else 0.0
        std_f0 = float(np.std(f0_clean)) if f0_clean.size > 0 else 0.0
        strength = float(np.nan_to_num(np.mean(voiced_probs))) if voiced_probs is not None else 0.0
        return np.array([mean_f0, std_f0, strength], dtype=float)
    except Exception:
        # 在 pyin 出错（例如信号太短/不含谐波）时退化为 0 向量
        return np.array([0.0, 0.0, 0.0], dtype=float)


# ---- 包络 ----
def calculate_amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    try:
        env = np.abs(hilbert(signal))
        return np.array([float(np.nan_to_num(np.mean(env))), float(np.nan_to_num(np.std(env)))])
    except Exception:
        return np.array([0.0, 0.0])


def calculate_envelope_peaks(signal: np.ndarray) -> float:
    try:
        env = np.abs(hilbert(signal))
        if len(env) == 0:
            return 0.0
        peaks, props = find_peaks(env, height=np.mean(env) if np.any(env) else 0.0)
        if len(peaks) == 0:
            return 0.0
        return float(np.nan_to_num(np.mean(env[peaks])))
    except Exception:
        return 0.0




# ---- 感知特征 ----
def calculate_loudness(signal: np.ndarray) -> float:
    return calculate_rms(signal)


def calculate_roughness(signal: np.ndarray) -> float:
    return float(np.nan_to_num(np.std(signal))) if len(signal) > 0 else 0.0


def calculate_timbre(signal: np.ndarray, sr: int) -> float:
    # 采用谱质心作为 timbre 的可解释替代
    return calculate_spectral_centroid(signal, sr)


def calculate_speech_interference_level(signal: np.ndarray) -> float:
    # 保持为 RMS 的代理（与原实现一致）
    return calculate_rms(signal)


# --------------------------
# 全局特征列表（确保名-值一致）
# --------------------------
FEATURE_LIST = [
    # 时域
    "energy", "zcr", "stam", "rms", "p2p",
    # 频域
    "centroid", "roll_off", "flatness",
    # 对比度（14: 7 means + 7 stds）
    *[f"contrast_{i}" for i in range(14)],
    # MFCC (12)
    *[f"mfcc_{i}" for i in range(12)],
    # delta-delta (12)
    *[f"delta2_mfcc_{i}" for i in range(12)],
    # 时频
    "stft",
    # pitch (3)
    "pitch_strength",
    # envelope
    "env_mean", "env_std", "env_peaks",
    # 感知
    "loudness", "roughness", "timbre", "speech_interf"
]


# --------------------------
# 主函数：imf_features（修正版）
# --------------------------
def IMF_features_list(U: np.ndarray,
                 fs: float,
                 phys_bands: Optional[List[Tuple[float, float]]] = None,
                 selected_features: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str]]:
    """
    U: np.ndarray shape (K, N) -- K 个 IMF，每个长度 N（允许不同长度也可，但你这里假设相同）
    fs: sampling rate
    返回: tokens (K x n_features) 以及 feat_names 列表
    """

    if selected_features is None:
        selected_features = FEATURE_LIST


    K = U.shape[0]

    tokens = []
    for k in range(K):
        u = U[k, :].astype(float)
        feats = {}

        # 时域
        feats["energy"] = calculate_energy(u)
        feats["zcr"] = calculate_zero_crossing_rate(u)
        feats["stam"] = calculate_short_time_avg_magnitude(u)
        feats["rms"] = calculate_rms(u)
        feats["p2p"] = calculate_peak_to_peak(u)

        # 频域（使用统一 n_fft）
        n_fft_spec = 2048
        feats["centroid"] = calculate_spectral_centroid(u, fs, n_fft=n_fft_spec)
        feats["roll_off"] = calculate_spectral_roll_off(u, fs, n_fft=n_fft_spec)
        feats["flatness"] = calculate_spectral_flatness(u, fs, n_fft=n_fft_spec)

        # 对比度（14）
        contrast_feats = calculate_spectral_contrast(u, fs, n_fft=n_fft_spec)
        for i, v in enumerate(contrast_feats):
            feats[f"contrast_{i}"] = float(v)

        # MFCCs / delta2
        mfcc_vals = calculate_mfcc(u, fs, n_mfcc=12)
        for i, v in enumerate(mfcc_vals):
            feats[f"mfcc_{i}"] = float(v)


        d2_vals = calculate_delta_delta_mfcc(u, fs, n_mfcc=12)
        for i, v in enumerate(d2_vals):
            feats[f"delta2_mfcc_{i}"] = float(v)

        # 时频
        feats["stft"] = calculate_stft_features(u, fs, n_fft=512)

        # pitch (3)
        pvals = calculate_pitch_features(u, fs)
        feats["pitch_strength_0"] = float(pvals[0])
        feats["pitch_strength_1"] = float(pvals[1])
        feats["pitch_strength_2"] = float(pvals[2])

        # envelope
        env_vals = calculate_amplitude_envelope(u)
        feats["env_mean"], feats["env_std"] = float(env_vals[0]), float(env_vals[1])
        feats["env_peaks"] = float(calculate_envelope_peaks(u))


        # 感知
        feats["loudness"] = calculate_loudness(u)
        feats["roughness"] = calculate_roughness(u)
        feats["timbre"] = calculate_timbre(u, fs)
        feats["speech_interf"] = calculate_speech_interference_level(u)

        # 按 selected_features 顺序提取（缺失 key 会抛出 KeyError，便于早发现 mismatch）
        token_vec = [feats[fname] if fname in feats else 0.0 for fname in selected_features]
        tokens.append(token_vec)

    tokens_arr = np.array(tokens, dtype=float)  # shape: (K, n_features)
    return tokens_arr, selected_features

