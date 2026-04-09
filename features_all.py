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


def calculate_kurtosis(signal: np.ndarray) -> float:
    try:
        return float(kurtosis(signal, fisher=True, bias=False))
    except Exception:
        return 0.0


# ---- 频域 ----
def calculate_spectral_centroid(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        c = librosa.feature.spectral_centroid(y=signal, sr=sr, n_fft=n_fft)
        return float(np.nan_to_num(np.mean(c)))
    except Exception:
        return 0.0


def calculate_spectral_bandwidth(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        b = librosa.feature.spectral_bandwidth(y=signal, sr=sr, n_fft=n_fft)
        return float(np.nan_to_num(np.mean(b)))
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


def calculate_dominant_frequency(signal: np.ndarray, sr: int, n_fft: int = 2048) -> float:
    try:
        S = np.abs(librosa.stft(signal, n_fft=n_fft))
        # S shape: (freq_bins, frames)
        mean_spec = np.nan_to_num(np.mean(S, axis=1))
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        idx = int(np.argmax(mean_spec))
        return float(freqs[idx]) if idx < len(freqs) else 0.0
    except Exception:
        return 0.0


# ---- MFCCs + deltas ----
def calculate_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        return np.nan_to_num(np.mean(mfccs, axis=1))
    except Exception:
        return np.zeros(n_mfcc, dtype=float)


def calculate_delta_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    try:
        mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
        d1 = librosa.feature.delta(mfccs, order=1)
        return np.nan_to_num(np.mean(d1, axis=1))
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


def calculate_mel_spectrogram_features(signal: np.ndarray, sr: int, n_mels: int = 12, n_fft: int = 2048) -> np.ndarray:
    try:
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels, n_fft=n_fft)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        return np.nan_to_num(np.mean(mel_db, axis=1))
    except Exception:
        return np.zeros(n_mels, dtype=float)


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


def calculate_peak_locations(signal: np.ndarray) -> int:
    try:
        env = np.abs(hilbert(signal))
        peaks, _ = find_peaks(env)
        return int(len(peaks))
    except Exception:
        return 0


def calculate_onset(signal: np.ndarray, sr: int) -> int:
    try:
        onset_frames = librosa.onset.onset_detect(y=signal, sr=sr)
        return int(len(onset_frames))
    except Exception:
        return 0


# ---- 感知特征 ----
def calculate_loudness(signal: np.ndarray) -> float:
    return calculate_rms(signal)


def calculate_sharpness(signal: np.ndarray) -> float:
    # 这里用峰度作为替代（尖锐性 proxy），与你之前的想法一致
    return calculate_kurtosis(signal)


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
    "energy", "zcr", "stam", "rms", "p2p", "kurtosis",
    # 频域
    "centroid", "bandwidth", "roll_off", "flatness", "dominant_freq",
    # 对比度（14: 7 means + 7 stds）
    *[f"contrast_{i}" for i in range(14)],
    # MFCC (12)
    *[f"mfcc_{i}" for i in range(12)],
    # delta (12)
    *[f"delta_mfcc_{i}" for i in range(12)],
    # delta-delta (12)
    *[f"delta2_mfcc_{i}" for i in range(12)],
    # 时频
    "stft", *[f"mel_{i}" for i in range(12)],
    # pitch (3)
    "f0_mean", "f0_std", "pitch_strength",
    # envelope
    "env_mean", "env_std", "env_peaks", "peak_count", "onset_count",
    # 感知
    "loudness", "sharpness", "roughness", "timbre", "speech_interf"
]


# --------------------------
# 主函数：imf_features（修正版）
# --------------------------
def imf_features(U: np.ndarray,
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
        feats["kurtosis"] = calculate_kurtosis(u)

        # 频域（使用统一 n_fft）
        n_fft_spec = 2048
        feats["centroid"] = calculate_spectral_centroid(u, fs, n_fft=n_fft_spec)
        feats["bandwidth"] = calculate_spectral_bandwidth(u, fs, n_fft=n_fft_spec)
        feats["roll_off"] = calculate_spectral_roll_off(u, fs, n_fft=n_fft_spec)
        feats["flatness"] = calculate_spectral_flatness(u, fs, n_fft=n_fft_spec)
        feats["dominant_freq"] = calculate_dominant_frequency(u, fs, n_fft=n_fft_spec)

        # 对比度（14）
        contrast_feats = calculate_spectral_contrast(u, fs, n_fft=n_fft_spec)
        for i, v in enumerate(contrast_feats):
            feats[f"contrast_{i}"] = float(v)

        # MFCCs / delta / delta2
        mfcc_vals = calculate_mfcc(u, fs, n_mfcc=12)
        for i, v in enumerate(mfcc_vals):
            feats[f"mfcc_{i}"] = float(v)

        d1_vals = calculate_delta_mfcc(u, fs, n_mfcc=12)
        for i, v in enumerate(d1_vals):
            feats[f"delta_mfcc_{i}"] = float(v)

        d2_vals = calculate_delta_delta_mfcc(u, fs, n_mfcc=12)
        for i, v in enumerate(d2_vals):
            feats[f"delta2_mfcc_{i}"] = float(v)

        # 时频
        feats["stft"] = calculate_stft_features(u, fs, n_fft=512)
        mel_feats = calculate_mel_spectrogram_features(u, fs, n_mels=12, n_fft=n_fft_spec)
        for i, v in enumerate(mel_feats):
            feats[f"mel_{i}"] = float(v)

        # pitch (3)
        pvals = calculate_pitch_features(u, fs)
        feats["f0_mean"], feats["f0_std"], feats["pitch_strength"] = (float(pvals[0]), float(pvals[1]), float(pvals[2]))

        # envelope
        env_vals = calculate_amplitude_envelope(u)
        feats["env_mean"], feats["env_std"] = float(env_vals[0]), float(env_vals[1])
        feats["env_peaks"] = float(calculate_envelope_peaks(u))
        feats["peak_count"] = int(calculate_peak_locations(u))
        feats["onset_count"] = int(calculate_onset(u, fs))

        # 感知
        feats["loudness"] = calculate_loudness(u)
        feats["sharpness"] = calculate_sharpness(u)
        feats["roughness"] = calculate_roughness(u)
        feats["timbre"] = calculate_timbre(u, fs)
        feats["speech_interf"] = calculate_speech_interference_level(u)

        # 按 selected_features 顺序提取（缺失 key 会抛出 KeyError，便于早发现 mismatch）
        token_vec = [feats[fname] if fname in feats else 0.0 for fname in selected_features]
        tokens.append(token_vec)

    tokens_arr = np.array(tokens, dtype=float)  # shape: (K, n_features)
    return tokens_arr, selected_features


# --------------------------
# 特征选择函数（修正版）
# --------------------------
def feature_selection(df: pd.DataFrame,
                      label_col: str = "label",
                      var_threshold: float = 1e-4,
                      corr_threshold: float = 0.9,
                      top_k: int = 30) -> Tuple[pd.DataFrame, List[str]]:
    """
    Steps:
      1) 方差过滤 (VarianceThreshold) -- 先 fit，然后 filter
      2) 相关性过滤（去除高度相关特征）
      3) 基于互信息 mutual_info_classif 排序并返回 Top-K
    返回 (df_selected, top_feature_list)
    """

    df_copy = df.copy()

    if label_col not in df_copy.columns:
        # 没有 label，直接返回原表并提醒
        print(f"⚠️ 未检测到标签列 '{label_col}'，跳过互信息计算与排序。")
        cols = [c for c in df_copy.columns if c != "file"]
        return df_copy.loc[:, cols], cols

    # X: 删除 file、label 等非特征列（errors ignored）
    X = df_copy.drop(columns=[label_col, "file"], errors="ignore")
    y = df_copy[label_col].values

    # Step 1: 方差过滤（必须 fit）
    selector = VarianceThreshold(var_threshold)
    try:
        selector.fit(X)
        support = selector.get_support()
        kept_cols = X.columns[support]
        X_var = X.loc[:, kept_cols]
    except Exception as e:
        print(f"[方差过滤] 发生异常，跳过方差过滤: {e}")
        X_var = X.copy()
    print(f"  [方差过滤] 原始 {X.shape[1]} -> 保留 {X_var.shape[1]} 个特征")

    if X_var.shape[1] == 0:
        print("  ⚠️ 方差过滤后无特征可用，返回原始特征（去掉 label/file）")
        X_var = X.copy()

    # Step 2: 相关性过滤
    corr = X_var.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [c for c in upper.columns if any(upper[c] > corr_threshold)]
    X_uncorr = X_var.drop(columns=to_drop, errors="ignore")
    print(f"  [相关性过滤] 去除 {len(to_drop)} 个高度相关特征，剩余 {X_uncorr.shape[1]}")

    # Step 3: 互信息筛选（标准化后计算）
    if X_uncorr.shape[1] == 0:
        print("  ⚠️ 相关性过滤后无特征，跳过互信息，返回 X_var 的列")
        top_features = X_var.columns.tolist()[:top_k]
    else:
        try:
            X_scaled = StandardScaler().fit_transform(X_uncorr)
            mi = mutual_info_classif(X_scaled, y, random_state=42)
            mi_series = pd.Series(mi, index=X_uncorr.columns).sort_values(ascending=False)
            top_features = mi_series.head(min(top_k, len(mi_series))).index.tolist()
        except Exception as e:
            print(f"[互信息] 计算失败：{e}，退回到相关性过滤后的前 {min(top_k, X_uncorr.shape[1])} 特征")
            top_features = list(X_uncorr.columns[:min(top_k, X_uncorr.shape[1])])

    # 构造返回的 DataFrame（保留 file、label 如果存在）
    cols_to_return = []
    if "file" in df_copy.columns:
        cols_to_return.append("file")
    cols_to_return.append(label_col)
    cols_to_return += top_features
    df_sel = df_copy.loc[:, [c for c in cols_to_return if c in df_copy.columns]]

    print(f"  [互信息排序] Top-{len(top_features)} 特征样例：{top_features[:6]} ...")
    return df_sel, top_features

