import librosa
from scipy.signal import  find_peaks
from typing import Dict
import os
import re
import numpy as np
import torch
import collections
from dataio import load_wav, list_wavs_with_labels
from preprocessing import preprocess
from segmentation import frame_signal, ola_reconstruct, ola_selfcheck
from rmsa_vmd import estimate_K_by_spectral_peaks, alpha_from_SE, admm_ul_rmsa_vmd
from IMF_postprocess import prune_merge_imfs, compute_weights, reconstruct
from models_attn_fuser import IMFTokenFuser
from train_classifier import crossval_train_eval
from metrics import bo_objective
from config import CFG
from pathlib import Path
from IMF_FBE_select import select_imfs_by_FBE

# ======  准备函数  ===========
def snr_db(x, xhat):
    err = x - xhat
    return 10.0 * np.log10((np.sum(x**2) + 1e-12) / (np.sum(err**2) + 1e-12))

def rmse(x, xhat):
    return float(np.sqrt(np.mean((x - xhat)**2)))

def energy_retention(s, shat):
    # ER = ||ŝ||^2 / ||s||^2
    return float((np.sum(shat**2) + 1e-12) / (np.sum(s**2) + 1e-12))

def full_reconstruct_from_U(U, e=None, add_residual=False):
    """
    U: (K, N)  未去噪的IMFs
    e: (N,)    残差（可选）
    add_residual: 是否把 e 加回
    """
    shat = np.sum(U, axis=0)
    if add_residual and (e is not None):
        shat = shat + e
    return shat

def vmd_consistency_and_rescale(U, e, s):
    """
    针对单段：测试三种组合并选最优，再做全局最小二乘增益校正。
      "+e": sum(U)+e
      "-e": e - sum(U)   # 防止你的 e 符号定义相反
      "Uonly": sum(U)    # 有的实现把 e 当纯噪声
    返回：修正后的 U,e、最优重构、调试信息。
    """
    sumU = np.sum(U, axis=0)
    cand = {"+e": sumU + e, "-e": e - sumU, "Uonly": sumU}
    snrs = {k: snr_db(s, v) for k, v in cand.items()}
    best = max(snrs, key=snrs.get)
    v0 = cand[best]

    # 全局增益（最小二乘）
    g = float(np.dot(s, v0) / (np.sum(v0**2) + 1e-12))
    v = g * v0

    # 按最优组合与增益，回写 U/e
    if best == "+e":
        U_fix, e_fix = g * U, g * e
    elif best == "-e":
        U_fix, e_fix = g * U, -g * e
    else:  # "Uonly"
        U_fix, e_fix = g * U, np.zeros_like(e)

    info = {
        "snr_cands": snrs,
        "best": best,
        "gain": g,
        "snr_after": snr_db(s, v),
        "er_after": energy_retention(s, v),
        "frac": {
            "sumU/seg": float(np.sum(sumU**2) / (np.sum(s**2) + 1e-12)),
            "e/seg": float(np.sum(e**2) / (np.sum(s**2) + 1e-12)),
        }
    }
    return U_fix, e_fix, v, info

def RMSA_vmd(seg, fs):
    # 1) K/alpha/lambda 自适应
    # ↑ 通过谱峰估计 IMF 数目，限定最大值为 CFG.vmd.K_max（需保证最小 >= 1）  rmsa_vmd.py
    K = estimate_K_by_spectral_peaks(seg, fs, CFG.vmd.K_max)

    # ↑ 依据谱熵（SE）调整 VMD 惩罚因子 alpha（噪声越强 alpha 越大，抑制带宽）  rmsa_vmd.py
    alpha = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)

    # ↑ 试图计算 MAD 估计噪声尺度，但这里**有个小 bug**：
    sigma = np.median(np.abs(seg - np.median(seg))) * 1.4826
    # ↑ 随机采样一个系数 c（区间来自配置），用于控制 L1 稀疏强度
    c = np.random.uniform(*CFG.vmd.lambda_c_range)
    # ↑ L1 正则的阈值（越大则更强稀疏），与噪声尺度成正比
    lam1 = c * sigma
    print(f"估计 IMF 数={K}, alpha={alpha:.4f}, lam1={lam1:.4f}")

    # 2) UL-RMSA-VMD (单段)：返回 U, W, e   rmsa_vmd.py
    # ↑ 采用自适应的 UL-RMSA-VMD 算法（ADMM 实现）
    #   输入：段信号 seg、采样率、K、alpha 以及各类正则系数和迭代参数
    #   输出：
    #     U: (L, N) 分解得到的 IMF 组（L ≤ K）
    #     W: 可能为瞬时频/带宽等结构化先验矩阵（取决于你的实现）
    #     e: (N,) 残差 / 噪声项
    U, W, e = admm_ul_rmsa_vmd(seg, fs, K, alpha,
                               lam1=lam1,
                               lam2=CFG.vmd.gamma_bw,
                               lam3=CFG.vmd.gamma_w_smooth,
                               lam4=CFG.vmd.gamma_lock,
                               iters=CFG.vmd.admm_iters, tol=CFG.vmd.admm_tol,
                               seed=CFG.vmd.seed)
    print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    return U, W, e, K, alpha, lam1


DEFAULT_SR = 16000
# -----------------------------------------------------------
# 1. 文件读取函数
# -----------------------------------------------------------

def load_imf_data(file_path: str) -> Dict[str, np.ndarray]:
    """加载指定路径的 npz 文件，返回包含 IMF 数据的字典"""
    try:
        with np.load(file_path, allow_pickle=True) as data:
            imfs = {k: v for k, v in data.items() if isinstance(v, np.ndarray) and v.size > 0}
            return imfs
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return {}


# -----------------------------------------------------------
# 2. 特征计算函数（按表格顺序）
# -----------------------------------------------------------

# ---- 时域信号 ----
def calculate_energy(signal: np.ndarray) -> float:
    return np.sum(signal ** 2)


def calculate_zero_crossing_rate(signal: np.ndarray) -> float:
    return np.sum(np.abs(np.diff(np.sign(signal)))) / len(signal)-1


def calculate_short_time_avg_magnitude(signal: np.ndarray) -> float:
    return np.mean(np.abs(signal))


def calculate_rms(signal: np.ndarray) -> float:
    return np.sqrt(np.mean(signal ** 2))


def calculate_peak_to_peak(signal: np.ndarray) -> float:
    return np.ptp(signal)


def calculate_kurtosis(signal: np.ndarray) -> float:
    return kurtosis(signal)


# ---- 频域信号 ----
def calculate_spectral_centroid(signal: np.ndarray, sr: int) -> float:
    return np.mean(librosa.feature.spectral_centroid(y=signal, sr=sr)[0])


def calculate_spectral_bandwidth(signal: np.ndarray, sr: int) -> float:
    return np.mean(librosa.feature.spectral_bandwidth(y=signal, sr=sr)[0])


def calculate_spectral_roll_off(signal: np.ndarray, sr: int) -> float:
    return np.mean(librosa.feature.spectral_rolloff(y=signal, sr=sr)[0])


def calculate_spectral_flatness(signal: np.ndarray, sr: int) -> float:
    return np.mean(librosa.feature.spectral_flatness(y=signal)[0])


def calculate_spectral_contrast(signal: np.ndarray, sr: int, n_bands: int = 6) -> np.ndarray:
    contrast = librosa.feature.spectral_contrast(y=signal, sr=sr)
    return np.concatenate([np.mean(contrast, axis=1), np.std(contrast, axis=1)])


def calculate_dominant_frequency(signal: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(signal))
    freqs = librosa.fft_frequencies(sr=sr)
    dominant_freq = freqs[np.argmax(np.mean(S, axis=1))]
    return dominant_freq


# ---- MFCCs ----
def calculate_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfccs, axis=1)


def calculate_delta_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 20) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta = librosa.feature.delta(mfccs)
    return np.mean(delta, axis=1)


def calculate_delta_delta_mfcc(signal: np.ndarray, sr: int, n_mfcc: int = 12) -> np.ndarray:
    mfccs = librosa.feature.mfcc(y=signal, sr=sr, n_mfcc=n_mfcc)
    delta2 = librosa.feature.delta(mfccs, order=2)
    return np.mean(delta2, axis=1)


# ---- 时频域 ----
def calculate_stft_features(signal: np.ndarray, sr: int) -> float:
    S = np.abs(librosa.stft(signal))
    return np.mean(S)


def calculate_mel_spectrogram_features(signal: np.ndarray, sr: int, n_mels: int = 12) -> np.ndarray:
    mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_mels=n_mels)
    return np.mean(mel, axis=1)


# ---- Pitch ----
def calculate_pitch_features(signal: np.ndarray, sr: int) -> np.ndarray:
    f0, _, voiced_probs = librosa.pyin(signal, fmin=50, fmax=1000, sr=sr)
    f0_clean = np.nan_to_num(f0)
    f0_mean = np.mean(f0_clean)
    f0_std = np.std(f0_clean)
    strength = np.mean(voiced_probs)
    return np.array([f0_mean, f0_std, strength])


# ---- 包络 ----
def calculate_amplitude_envelope(signal: np.ndarray) -> np.ndarray:
    env = np.abs(hilbert(signal))
    return np.array([np.mean(env), np.std(env)])


def calculate_envelope_peaks(signal: np.ndarray) -> float:
    env = np.abs(hilbert(signal))
    peaks, _ = find_peaks(env, height=np.mean(env))
    return np.mean(env[peaks]) if len(peaks) > 0 else 0.0


def calculate_peak_locations(signal: np.ndarray) -> int:
    env = np.abs(hilbert(signal))
    peaks, _ = find_peaks(env)
    return len(peaks)


def calculate_onset(signal: np.ndarray, sr: int) -> int:
    onset_frames = librosa.onset.onset_detect(y=signal, sr=sr)
    return len(onset_frames)


# ---- 感知特征 ----
def calculate_loudness(signal: np.ndarray) -> float:
    return calculate_rms(signal)


def calculate_sharpness(signal: np.ndarray) -> float:
    return calculate_kurtosis(signal)


def calculate_roughness(signal: np.ndarray) -> float:
    return np.std(signal)


def calculate_timbre(signal: np.ndarray) -> float:
    return np.mean(np.abs(np.diff(signal)))


def calculate_speech_interference_level(signal: np.ndarray) -> float:
    return calculate_rms(signal)


# -----------------------------------------------------------
# 3. 综合特征提取主函数
# -----------------------------------------------------------

FEATURE_LIST = [
    # 时域
    "energy", "zcr", "stam", "rms", "p2p", "kurtosis",
    # 频域
    "centroid", "bandwidth", "roll_off", "flatness", "dominant_freq",
    # 对比度
    *[f"contrast_{i}" for i in range(14)],  # 7均值+7std
    # MFCC
    *[f"mfcc_{i}" for i in range(12)],
    *[f"delta_mfcc_{i}" for i in range(12)],
    *[f"delta2_mfcc_{i}" for i in range(12)],
    # 时频
    "stft", *[f"mel_{i}" for i in range(12)],
    # Pitch
    "f0_mean", "f0_std", "pitch_strength",
    # 包络
    "env_mean", "env_std", "env_peaks", "peak_count", "onset_count",
    # 感知特征
    "loudness", "sharpness", "roughness", "timbre", "speech_interf"
]

import numpy as np
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis
from typing import List, Tuple

def imf_features(U: np.ndarray, fs: float, selected_features: list = None, contrast_n_bands: int = 6):
    K, N = U.shape
    tokens = []

    # 默认提取所有特征
    if selected_features is None:
        selected_features = FEATURE_LIST

    for k in range(K):
        u = U[k]
        feats = {}
        # 时域
        feats["energy"] = calculate_energy(u)
        feats["zcr"] = calculate_zero_crossing_rate(u)
        feats["stam"] = calculate_short_time_avg_magnitude(u)
        feats["rms"] = calculate_rms(u)
        feats["p2p"] = calculate_peak_to_peak(u)
        feats["kurtosis"] = calculate_kurtosis(u)

        # 频域
        feats["centroid"] = calculate_spectral_centroid(u, fs)
        feats["bandwidth"] = calculate_spectral_bandwidth(u, fs)
        feats["roll_off"] = calculate_spectral_roll_off(u, fs)
        feats["flatness"] = calculate_spectral_flatness(u, fs)
        feats["dominant_freq"] = calculate_dominant_frequency(u, fs)

        # 对比度
        contrast_feats = calculate_spectral_contrast(u, fs, n_bands=contrast_n_bands)
        for i, v in enumerate(contrast_feats):
            feats[f"contrast_{i}"] = v

        # MFCC
        for i, v in enumerate(calculate_mfcc(u, fs)):
            feats[f"mfcc_{i}"] = v
        for i, v in enumerate(calculate_delta_mfcc(u, fs)):
            feats[f"delta_mfcc_{i}"] = v
        for i, v in enumerate(calculate_delta_delta_mfcc(u, fs)):
            feats[f"delta2_mfcc_{i}"] = v

        # 时频
        feats["stft"] = calculate_stft_features(u, fs)
        mel_feats = calculate_mel_spectrogram_features(u, fs)
        for i, v in enumerate(mel_feats):
            feats[f"mel_{i}"] = v

        # Pitch
        pitch_vals = calculate_pitch_features(u, fs)
        feats["f0_mean"], feats["f0_std"], feats["pitch_strength"] = pitch_vals

        # 包络
        env_vals = calculate_amplitude_envelope(u)
        feats["env_mean"], feats["env_std"] = env_vals
        feats["env_peaks"] = calculate_envelope_peaks(u)
        feats["peak_count"] = calculate_peak_locations(u)
        feats["onset_count"] = calculate_onset(u, fs)

        # 感知特征
        feats["loudness"] = calculate_loudness(u)
        feats["sharpness"] = calculate_sharpness(u)
        feats["roughness"] = calculate_roughness(u)
        feats["timbre"] = calculate_timbre(u)
        feats["speech_interf"] = calculate_speech_interference_level(u)

        # 按 selected_features 顺序提取
        token_vec = [feats[fname] for fname in selected_features]
        tokens.append(np.array(token_vec, dtype=float))

    return np.vstack(tokens)

# ======  准备函数  ===========
def snr_db(x, xhat):
    err = x - xhat
    return 10.0 * np.log10((np.sum(x**2) + 1e-12) / (np.sum(err**2) + 1e-12))

def rmse(x, xhat):
    return float(np.sqrt(np.mean((x - xhat)**2)))

def energy_retention(s, shat):
    # ER = ||ŝ||^2 / ||s||^2
    return float((np.sum(shat**2) + 1e-12) / (np.sum(s**2) + 1e-12))

def full_reconstruct_from_U(U, e=None, add_residual=False):
    """
    U: (K, N)  未去噪的IMFs
    e: (N,)    残差（可选）
    add_residual: 是否把 e 加回
    """
    shat = np.sum(U, axis=0)
    if add_residual and (e is not None):
        shat = shat + e
    return shat

def vmd_consistency_and_rescale(U, e, s):
    """
    针对单段：测试三种组合并选最优，再做全局最小二乘增益校正。
      "+e": sum(U)+e
      "-e": e - sum(U)   # 防止你的 e 符号定义相反
      "Uonly": sum(U)    # 有的实现把 e 当纯噪声
    返回：修正后的 U,e、最优重构、调试信息。
    """
    sumU = np.sum(U, axis=0)
    cand = {"+e": sumU + e, "-e": e - sumU, "Uonly": sumU}
    snrs = {k: snr_db(s, v) for k, v in cand.items()}
    best = max(snrs, key=snrs.get)
    v0 = cand[best]

    # 全局增益（最小二乘）
    g = float(np.dot(s, v0) / (np.sum(v0**2) + 1e-12))
    v = g * v0

    # 按最优组合与增益，回写 U/e
    if best == "+e":
        U_fix, e_fix = g * U, g * e
    elif best == "-e":
        U_fix, e_fix = g * U, -g * e
    else:  # "Uonly"
        U_fix, e_fix = g * U, np.zeros_like(e)

    info = {
        "snr_cands": snrs,
        "best": best,
        "gain": g,
        "snr_after": snr_db(s, v),
        "er_after": energy_retention(s, v),
        "frac": {
            "sumU/seg": float(np.sum(sumU**2) / (np.sum(s**2) + 1e-12)),
            "e/seg": float(np.sum(e**2) / (np.sum(s**2) + 1e-12)),
        }
    }
    return U_fix, e_fix, v, info

def RMSA_vmd(seg, fs):
    # 1) K/alpha/lambda 自适应
    # ↑ 通过谱峰估计 IMF 数目，限定最大值为 CFG.vmd.K_max（需保证最小 >= 1）  rmsa_vmd.py
    K = estimate_K_by_spectral_peaks(seg, fs, CFG.vmd.K_max)

    # ↑ 依据谱熵（SE）调整 VMD 惩罚因子 alpha（噪声越强 alpha 越大，抑制带宽）  rmsa_vmd.py
    alpha = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)

    # ↑ 试图计算 MAD 估计噪声尺度，但这里**有个小 bug**：
    sigma = np.median(np.abs(seg - np.median(seg))) * 1.4826
    # ↑ 随机采样一个系数 c（区间来自配置），用于控制 L1 稀疏强度
    c = np.random.uniform(*CFG.vmd.lambda_c_range)
    # ↑ L1 正则的阈值（越大则更强稀疏），与噪声尺度成正比
    lam1 = c * sigma
    print(f"估计 IMF 数={K}, alpha={alpha:.4f}, lam1={lam1:.4f}")

    # 2) UL-RMSA-VMD (单段)：返回 U, W, e   rmsa_vmd.py
    # ↑ 采用自适应的 UL-RMSA-VMD 算法（ADMM 实现）
    #   输入：段信号 seg、采样率、K、alpha 以及各类正则系数和迭代参数
    #   输出：
    #     U: (L, N) 分解得到的 IMF 组（L ≤ K）
    #     W: 可能为瞬时频/带宽等结构化先验矩阵（取决于你的实现）
    #     e: (N,) 残差 / 噪声项
    U, W, e = admm_ul_rmsa_vmd(seg, fs, K, alpha,
                               lam1=lam1,
                               lam2=CFG.vmd.gamma_bw,
                               lam3=CFG.vmd.gamma_w_smooth,
                               lam4=CFG.vmd.gamma_lock,
                               iters=CFG.vmd.admm_iters, tol=CFG.vmd.admm_tol,
                               seed=CFG.vmd.seed)
    print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    return U, W, e, K, alpha, lam1

def extract_segment_features(seg, fs, phys_bands):
    """
        对单个振动段 seg 执行完整特征提取流程：
          1. RMSA-VMD 分解
          2. IMF 一致性修正
          3. IMF 剪枝 / FBE 筛选
          4. 权重计算 + 去噪重构
          5. IMF-token 特征提取及保存
    """
    # 1) RMSA-VMD 分解
    U, W, e, K, alpha, lam1 = RMSA_vmd(seg, fs)
    U, e, rec_dbg, dbg = vmd_consistency_and_rescale(U, e, seg)
    print(f"  [一致性] 试探SNR={ {k: f'{v:.2f}' for k, v in dbg['snr_cands'].items()} } "
          f"=> best={dbg['best']}, gain={dbg['gain']:.3f}, "
          f"SNR*={dbg['snr_after']:.2f} dB, ER*={dbg['er_after']:.3f}")
    print(f"  [能量分配] ||sumU||^2/||seg||^2={dbg['frac']['sumU/seg']:.3f}, "
          f"||e||^2/||seg||^2={dbg['frac']['e/seg']:.3f}")

    # 2) 模态剪枝与去噪权重
    if getattr(CFG.post, "use_fbe", False):
        U, fbe_vals = select_imfs_by_FBE(U, fs, use_envelope=True, top_k=CFG.post.fbe_top_k)
        print(f"  [FBE筛选] 保留IMF数={U.shape[0]}")
    else:
        U = prune_merge_imfs(U, CFG.post.prune_energy_thr, CFG.post.prune_corr_thr)
        print(f"  [标准剪枝] IMF数={U.shape[0]}")

    # 3) 权重与去噪重构
    s_sum = U.sum(axis=0)
    a_sum = np.dot(seg, s_sum) / (np.dot(s_sum, s_sum) + 1e-12)
    s_sum = a_sum * s_sum
    print(f"  [BASELINE sum(U) after LS] SNR={snr_db(seg, s_sum):.2f} dB, RMSE={rmse(seg, s_sum):.4f}")
    s_sum_e = s_sum + e  # 这里 e 很小，按需保留
    print(f"  [BASELINE sum(U)+e after LS] SNR={snr_db(seg, s_sum_e):.2f} dB, RMSE={rmse(seg, s_sum_e):.4f}")

    w = compute_weights(U, e, CFG.post.weight_eta)
    w = np.asarray(w, float)
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    s_hat, U_den = reconstruct(U, e, w, CFG.post.wavelet, CFG.post.wavelet_levels)
    print(f"  去噪重构完成: U_den.shape={U_den.shape}, s_hat.shape={s_hat.shape}")

    # RMS增益对齐
    # 线性合成
    #s_tilde = (w[:, None] * U_den).sum(axis=0)  # 是否 + e 取决于你的定义
    rms_seg = np.sqrt(np.mean(seg ** 2))
    rms_tilde = np.sqrt(np.mean(s_hat ** 2))
    a = 1.0 if rms_tilde < 1e-12 else rms_seg / rms_tilde
    s_hat = a * s_hat

    # 4）指标
    snr_val = snr_db(seg, s_hat)
    rmse_val = rmse(seg, s_hat)
    er_val = float(np.sum(s_hat ** 2) / (np.sum(seg ** 2) + 1e-12))
    corr = float(np.corrcoef(seg, s_hat)[0, 1])
    print(f"  重构指标(对齐后): SNR={snr_val:.2f} dB, RMSE={rmse_val:.4f}, ER={er_val:.3f}, corr={corr:.4f}")

    # 5) IMF-token 特征
    # tokens_1 = imf_features(U_den, fs, phys_bands)
    # tokens_2 = []
    # for k in range(U_den.shape[0]):
    #     mfcc_vec = mfcc_features(U_den[k], fs, n=20, use_deltas=True)
    #     tokens_2.append(mfcc_vec)
    # tokens_2 = np.vstack(tokens_2)
    # tokens = np.hstack([tokens_1, tokens_2])

    # ===提取单个特征
    #tokens = imf_features(U, fs, selected_features=[f"contrast_{i}" for i in range(14)], contrast_n_bands=6)
    tokens = imf_features(U, fs, selected_features=[f"loudness"])

    # ===提取多个特征
    # features_to_use = ["energy", "rms", "centroid", "f0_mean"]
    # tokens = imf_features(U, fs, selected_features=features_to_use)

    # 6） 中心频率
    f_center = []
    for k in range(U_den.shape[0]):
        Uk = np.fft.rfft(U_den[k])
        freqs = np.fft.rfftfreq(len(U_den[k]), 1 / fs)
        P = np.abs(Uk) ** 2 + 1e-12
        f_center.append((freqs * P).sum() / P.sum())
    f_center = np.array(f_center, dtype=float)

    return tokens, f_center, s_hat, (U_den, w, e), snr_val, rmse_val


# ======  分解阶段  ===========
def step1_vmd_decompose(data_root: str, vmd_save_dir: str, single_file: str = None):
    """
       仅执行VMD分解，将每段的IMF组、残差、原始信号等保存到vmd_save_dir
    """
    os.makedirs(vmd_save_dir, exist_ok=True)
    # 1. 加载数据列表
    if single_file is not None:
        items = [(single_file, 0)]
        print(f"[单文件模式] 处理文件: {single_file}")
    else:
        items = list_wavs_with_labels(data_root)
        assert len(items) > 0, "数据目录为空或未按预期组织"
        print(f"[批量模式] 共加载 {len(items)} 个文件")

    fs = CFG.seg.fs_target  # 目标采样率

    #  2. 逐个文件处理：预处理→分帧→VMD分解→保存
    for i, (path, y) in enumerate(items, 1):
        wav_stem = Path(path).stem
        # VMD结果保存路径（每个文件一个npz）
        vmd_save_path = os.path.join(vmd_save_dir, f"{wav_stem}_label{y}_vmd.npz")

        if os.path.exists(vmd_save_path):
            print(f"⚙️ 跳过已存在VMD结果: {vmd_save_path}")
            continue

        print(f"\n--- VMD分解文件 {i}/{len(items)}: {path} ---")
        # ↑ 读取 wav 并重采样到 fs
        x_raw = load_wav(path, fs)
        # ↑ 去直流、归一化、可选降采样/带通等
        x = preprocess(x_raw)
        # ↑ 分帧
        frames, idxs, win, hop = frame_signal(x, fs, CFG.seg.seg_len_s, CFG.seg.hop_ratio, CFG.seg.window)
        print(f"分帧: 共 {len(frames)} 段, frame_len={frames.shape[1]}, hop={hop}")

        seg_vmd_results = []
        for seg_id, seg in enumerate(frames):
            # 执行 VMD 分解部分
            print(f"  — 分解第 {seg_id + 1}/{len(frames)} 段 —")
            U, W, e, K, alpha, lam1 = RMSA_vmd(seg, fs)
            # 一致性修正 + 增益对齐
            U_fix, e_fix, v, info = vmd_consistency_and_rescale(U, e, seg)
            # 保存当前段的VMD结果
            seg_vmd_results.append({"U_fix": U_fix, "e_fix": e_fix, "seg": seg})

        # === 拼接所有帧的 IMF ===
        if len(seg_vmd_results) > 0:
            all_U = [res["U_fix"] for res in seg_vmd_results]
            K = max(U.shape[0] for U in all_U)
            frame_len = all_U[0].shape[1]
            hop = hop  # 来自 frame_signal
            total_len = frame_len + (len(all_U) - 1) * hop

            full_imf_group = np.zeros((K, total_len))
            weight_sum = np.zeros(total_len)
            window = win  # Hann窗或其他

            for seg_id, U in enumerate(all_U):
                start = seg_id * hop
                end = start + frame_len
                k_i, _ = U.shape
                # 加窗叠加
                full_imf_group[:k_i, start:end] += U * window[None, :]
                weight_sum[start:end] += window

            # 归一化（避免重叠区能量放大）
            full_imf_group /= (weight_sum + 1e-12)

            # 每个 IMF 求和后重构整体信号
            full_reconstructed = np.sum(full_imf_group, axis=0)

            print(f"[IMF组] 形状: {full_imf_group.shape} (模式数×总长度)")
            print(f"[信号重构] 形状: {full_reconstructed.shape}")
        else:
            full_imf_group = None
            full_reconstructed = None
            print("⚠️ 无有效 IMF 数据。")

        # 保存当前文件的所有VMD结果
        np.savez(
            vmd_save_path,
            wav_path=path,
            label=y,
            x_raw=x_raw,   # 原始音频/经由重采样，但是这里好像没有
            x_prep=x,      # 预处理后
            frame_win=win,
            frame_hop=hop,
            seg_vmd_results=np.array(seg_vmd_results, dtype=object),
            concatenated_U=full_reconstructed    # 分解后IMF简单拼接的重构信号
        )
        print(f"✅ 保存VMD结果: {vmd_save_path}")

# ======  特征提取  ===========
def step2_extract_features(vmd_save_dir: str, feat_save_dir: str):
    """
       从VMD结果目录读取数据，提取特征，按类别保存到feat_save_dir
    """
    os.makedirs(feat_save_dir, exist_ok=True)

    # 1. 加载所有VMD结果文件
    vmd_files = [f for f in os.listdir(vmd_save_dir) if f.endswith("_vmd.npz")]
    assert len(vmd_files) > 0, f"未找到VMD结果文件，请检查 {vmd_save_dir}"
    print(f"共加载 {len(vmd_files)} 个VMD结果文件")

    fs = CFG.seg.fs_target  # 与VMD阶段一致的采样率
    # 轻量注意力融合器
    fuser = IMFTokenFuser(d_model=CFG.fuser.d_model, nhead=CFG.fuser.nhead,
                          num_layers=CFG.fuser.num_layers, dropout=CFG.fuser.dropout,
                          se_ratio=CFG.fuser.se_ratio, bands=CFG.phys.bands)
    fuser.eval()
    proj = None  # 延迟初始化

    # 2. 逐个VMD结果文件处理：提取特征→按类别保存
    for f in vmd_files:
        vmd_file_path = os.path.join(vmd_save_dir, f)
        # 解析文件名中的标签和原始文件名
        label_match = re.search(r"label(\d+)_vmd\.npz", f)
        assert label_match, f"VMD文件名格式错误: {f}"
        label = int(label_match.group(1))
        wav_stem = f.replace(f"_label{label}_vmd.npz", "")

        # 特征保存路径（按类别分文件夹）
        class_dir = os.path.join(feat_save_dir, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        feat_save_path = os.path.join(class_dir, f"{wav_stem}_label{label}_feat.npz")

        if os.path.exists(feat_save_path):
            print(f"⚙️ 跳过已存在特征: {feat_save_path}")
            continue

        print(f"\n--- 处理VMD文件: {f}, 标签={label} ---")
        # 加载VMD结果
        vmd_data = np.load(vmd_file_path, allow_pickle=True)
        seg_vmd_results = vmd_data["seg_vmd_results"].tolist()
        frame_win = vmd_data["frame_win"]
        frame_hop = vmd_data["frame_hop"]
        print(f"  共 {len(seg_vmd_results)} 段VMD数据")

        # 逐个段提取特征
        seg_feats = []  # 存储每段的融合特征
        seg_attn = []  # 存储每段的注意力权重
        s_hat_list = []  # 存储每段的重构信号（供评估用）
        all_W_centers = []
        U_dens_all = []

        for seg_idx, seg_vmd in enumerate(seg_vmd_results):
            seg = seg_vmd["seg"]
            tokens, f_center, s_hat, (U_den, w, e), snr_val, rmse_val = extract_segment_features(seg, fs, CFG.phys.bands)

            s_hat_list.append(s_hat)
            U_dens_all.append(U_den)

            # ===== 融合处理 =====
            if proj is None:
                d_token = tokens.shape[1]
                proj = torch.nn.Linear(d_token, CFG.fuser.d_model)
                print(f"  自动初始化特征投影层: Linear({d_token}, {CFG.fuser.d_model})")

            T = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
            T = proj(T)
            F = torch.tensor(f_center, dtype=torch.float32).view(1, -1, 1)

            with torch.no_grad():
                z, h = fuser(T, F)  # z: 融合特征, h: 注意力权重

            seg_feats.append(z.numpy().squeeze(0))
            seg_attn.append(h.mean(dim=-1).squeeze(0).numpy())
            all_W_centers.append(f_center)
            print(f"  段{seg_idx + 1}: 特征维度={z.shape[1]}, 注意力权重数={h.shape[1]}")

         # 3. 计算文件级特征（所有段特征的均值）
        file_feat = np.mean(np.vstack(seg_feats), axis=0)
        # 全文件重构信号（供评估用）
        s_hat_frames = np.stack(s_hat_list, axis=0)
        s_hat_full = ola_reconstruct(s_hat_frames, frame_win, frame_hop, frames_are_windowed=True)
        print(f"[OLA check] frames={s_hat_frames.shape}, win={frame_win.shape}, hop={frame_hop}, "
              f"min(wsum-like)≈{np.min(np.convolve(frame_win, np.ones_like(frame_win))[::frame_hop]) if frame_hop > 0 else 'N/A'}")

        # 调试信息
        print(f"[调试] U_dens_all 长度: {len(U_dens_all)}")
        for i, U in enumerate(U_dens_all):
            print(f"  U[{i}].shape = {U.shape}")
        # === 拼接所有帧的 IMF ===
        if len(U_dens_all) > 0:
            # 重构信号
            reconstructed_segments = [np.sum(U, axis=0) for U in U_dens_all]
            full_reconstructed = np.concatenate(reconstructed_segments)

            # IMF矩阵
            max_k = max(U.shape[0] for U in U_dens_all)
            U_dens_padded = []
            for U in U_dens_all:
                K, N = U.shape
                if K < max_k:
                    pad = np.zeros((max_k - K, N))
                    U = np.vstack([U, pad])
                U_dens_padded.append(U)
            full_imf_matrix = np.concatenate(U_dens_padded, axis=1)

            print(f"[信号重构] 重构信号: {full_reconstructed.shape}")
            print(f"[IMF分析] IMF矩阵: {full_imf_matrix.shape}")

        # 4. 保存特征结果（按类别目录）
        np.savez(
            feat_save_path,
            wav_stem=wav_stem,
            label=label,
            file_feat=file_feat,  # 文件级融合特征
            seg_feats=seg_feats,  # 各段特征列表
            seg_attn=np.array(seg_attn, dtype=object), # 各段注意力权重列表
            s_hat_full=s_hat_full,
            U_den=full_reconstructed,
            full_imf_matrix=full_imf_matrix
        )
        print(f"✅ 保存 {feat_save_dir}")

# ======  训练验证  ===========
def train_model(model_name="svm", cv_folds=None, feature_dir="outputs/files_by_class"):
    files = sorted(Path(feature_dir).rglob("*.npz"))
    assert files, f"未发现特征文件，请确认 {feature_dir} 是否存在。"

    all_Z, all_Y = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_Z.append(d["file_feat"])
        all_Y.append(int(d["label"]))
    X = np.vstack(all_Z)
    y = np.array(all_Y)
    print(f"[加载训练数据] X.shape={X.shape}, y.shape={y.shape}")
    print("[类别分布]", collections.Counter(y))

    if cv_folds is None:
        cv_folds = CFG.bayes.cv_folds

    f1, aurc, y_true, y_pred, clf = crossval_train_eval(
        X, y, model_name=model_name, n_splits=cv_folds, return_preds=True
    )

    score = bo_objective(f1, aurc)
    print(f"[{model_name.upper()}] F1={f1:.4f}, AURC={aurc:.4f}, Objective={score:.4f}")

    from sklearn.metrics import classification_report, confusion_matrix
    print("\n分类报告：")
    print(classification_report(y_true, y_pred))
    print("混淆矩阵：\n", confusion_matrix(y_true, y_pred))

    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix ({model_name.upper()})")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    return f1, aurc, score, y_true, y_pred, clf

def step3_train_classifier(feature_dir="outputs/features", model_name="rf"):
    return train_model(model_name=model_name, feature_dir=feature_dir)

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def visualize_features(X, y, method="tsne", feature_names=None):
    if method == "pca":
        reducer = PCA(n_components=2)
    else:
        reducer = TSNE(n_components=2, perplexity=30, learning_rate='auto', init='pca')
    X_2d = reducer.fit_transform(X)

    plt.figure(figsize=(8,6))
    plt.scatter(X_2d[:,0], X_2d[:,1], c=y, cmap='tab10', alpha=0.7)
    plt.title(f"{method.upper()} visualization of features")
    plt.show()


from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
import numpy as np


def select_features(X, y, method="variance", top_k=10):
    if method == "variance":
        selector = VarianceThreshold(threshold=np.percentile(np.var(X, axis=0), 50))
        X_new = selector.fit_transform(X)
        return X_new, selector.get_support(indices=True)

    elif method == "correlation":
        corr = np.abs(np.corrcoef(X.T, y)[:-1, -1])
        idx = np.argsort(corr)[-top_k:]
        return X[:, idx], idx

    elif method == "mutual_info":
        mi = mutual_info_classif(X, y)
        idx = np.argsort(mi)[-top_k:]
        return X[:, idx], idx


# ======  主程序  ===========
if __name__ == "__main__":
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"

    # === 阶段1：分解 ===
    #step1_vmd_decompose(data_root, vmd_save_dir="outputs/imfs_3", single_file=None)

    # === 阶段2：特征提取 ===
    step2_extract_features(vmd_save_dir="outputs/imfs_2", feat_save_dir="outputs/features_loudness")

    # === 阶段3：分类 ===
    step3_train_classifier(feature_dir="outputs/", model_name="rf")
