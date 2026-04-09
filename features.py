# features.py
# 特征工程
import numpy as np
import librosa
from scipy.stats import kurtosis
from signal_utils import spectral_entropy
from typing import Dict, List
import numpy as np
from scipy.signal import hilbert, welch
from scipy.stats import kurtosis
from typing import List, Tuple

def teager(x):
    x = np.asarray(x)
    y = np.zeros_like(x)
    y[1:-1] = x[1:-1]**2 - x[0:-2]*x[2:]
    return np.mean(np.abs(y))

def spectral_flatness(X):
    # X: magnitude spectrum
    gm = np.exp(np.mean(np.log(X+1e-12)))
    am = np.mean(X+1e-12)
    return gm/(am+1e-12)

def spectral_rolloff(x, fs, roll=0.85, nfft=2048):
    S = np.abs(np.fft.rfft(x, n=nfft))**2
    csum = np.cumsum(S)
    thr = roll*csum[-1]
    idx = np.searchsorted(csum, thr)
    freqs = np.fft.rfftfreq(nfft, 1/fs)
    return freqs[min(idx, len(freqs)-1)]

def pseudo_cyclic_features(x, fs, bands=[(50,70),(300,340)] , alpha=0.1):
    # 轻量“循环平稳”代理：在候选循环频带上用多窗平均能量与选择性指标
    feats = []
    total = np.sum(np.abs(np.fft.rfft(x))**2)+1e-12
    for lo, hi in bands:
        e = band_energy_ratio(x, fs, (lo,hi))
        feats += [e, e/(1-e+1e-12)]
    return np.array(feats, dtype=float)

def band_energy_ratio(x, fs, band):
    S = np.abs(np.fft.rfft(x))**2
    freqs = np.fft.rfftfreq(len(x), 1/fs)
    m = (freqs>=band[0]) & (freqs<=band[1])
    return float(np.sum(S[m])/(np.sum(S)+1e-12))

def mfcc_features(x, fs, n=20, use_deltas=True):
    mf = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n)
    feats = [np.mean(mf, axis=1), np.std(mf, axis=1)]
    if use_deltas:
        d1 = librosa.feature.delta(mf)
        d2 = librosa.feature.delta(mf, order=2)
        feats += [np.mean(d1,axis=1), np.std(d1,axis=1), np.mean(d2,axis=1), np.std(d2,axis=1)]
    return np.concatenate([f.ravel() for f in feats])

def _psd_feats(x, fs, nperseg=512):
    f, Pxx = welch(x, fs=fs, nperseg=min(nperseg, len(x)))
    Pxx = np.maximum(Pxx, 1e-18)
    P = Pxx / np.trapz(Pxx, f)           # 归一化功率密度
    f_cent = np.trapz(f * P, f)          # 频谱质心
    f_bw2 = np.trapz(((f - f_cent)**2) * P, f)  # 二阶矩 -> 带宽^2
    f_bw = np.sqrt(max(f_bw2, 0.0))
    return f_cent, f_bw

def _envelope_feats(x, fs, peak_search=(0.5, 200.0)):
    # Hilbert 包络
    env = np.abs(hilbert(x))
    # 包络统计
    e_mean = float(np.mean(env))
    e_std  = float(np.std(env) + 1e-12)
    e_kurt = float(kurtosis(env, fisher=False))
    # 包络谱主峰与 SNR
    nfft = 1 << int(np.ceil(np.log2(len(env))))
    E = np.abs(np.fft.rfft(env - env.mean(), n=nfft))**2
    freqs = np.fft.rfftfreq(nfft, 1.0/fs)
    m = (freqs >= peak_search[0]) & (freqs <= peak_search[1])
    if not np.any(m):
        return e_mean, e_std, e_kurt, 0.0, 0.0
    Es = E[m]; fsb = freqs[m]
    pk_idx = int(np.argmax(Es))
    pk_amp = float(Es[pk_idx])
    pk_freq = float(fsb[pk_idx])
    # 噪底取中位数（去掉 ±2 个 bin 的峰邻域）
    nb = 2
    nb_mask = np.ones_like(Es, dtype=bool)
    i0 = max(pk_idx - nb, 0); i1 = min(pk_idx + nb + 1, Es.size)
    nb_mask[i0:i1] = False
    noise_floor = float(np.median(Es[nb_mask])) if np.any(nb_mask) else 1e-12
    pk_snr = float(pk_amp / (noise_floor + 1e-12))
    # 稳健缩放
    return e_mean, e_std, e_kurt, pk_freq, np.log1p(pk_snr)

def _band_energy_ratio(x, fs, band: Tuple[float,float]):
    X = np.abs(np.fft.rfft(x))**2
    f = np.fft.rfftfreq(len(x), 1/fs)
    m = (f >= band[0]) & (f <= band[1])
    num = float(np.sum(X[m]))
    den = float(np.sum(X) + 1e-12)
    return num / den
def imf_features(U: np.ndarray, fs: float, phys_bands: List[Tuple[float,float]]):
    """
    强化版 IMF-token：
      [能量占比, PSD质心, PSD带宽, 包络均值, 包络Std, 包络峭度,
       包络谱峰频, 包络峰SNR(log1p), IF中位数, IF Std,
       谱平坦度旧特征(可选), rolloff(可选), 各物理频带能量比 ...]
    返回: (K, d_token)
    """
    K, N = U.shape
    tokens = []
    seg_energy = float(np.sum(U**2))

    for k in range(K):
        u = U[k]
        # 1) IMF 能量占比
        ek = float(np.sum(u**2)) / (seg_energy + 1e-12)

        # 2) PSD 质心/带宽（Welch，更稳）
        f_cent, f_bw = _psd_feats(u, fs)

        # 3) 包络域：统计 + 主峰 + 峰SNR
        e_mean, e_std, e_kurt, pkf, pksnr = _envelope_feats(u, fs, peak_search=(0.5, fs/4))

        # 4) 瞬时频率（FM指标）
        hu = hilbert(u)
        phase = np.unwrap(np.angle(hu))
        if_inst = (np.diff(phase) / (2*np.pi) * fs)
        if_inst = if_inst[np.isfinite(if_inst)]
        if if_inst.size == 0:
            if_med, if_std = 0.0, 0.0
        else:
            if_med = float(np.median(if_inst))
            if_std = float(np.std(if_inst))

        # 5) 物理频带能量指纹
        band_feats = [_band_energy_ratio(u, fs, b) for b in phys_bands]

        # 6) 稳健缩放/裁剪
        feat_vec = [
            np.log1p(max(ek, 1e-12)),
            f_cent, f_bw,
            e_mean, e_std, np.clip(e_kurt, 0.0, 100.0),
            pkf, pksnr,
            if_med, if_std,
        ] + band_feats

        tokens.append(np.array(feat_vec, dtype=float))

    return np.vstack(tokens)
