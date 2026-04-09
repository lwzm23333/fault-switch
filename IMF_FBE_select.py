# -*- coding: utf-8 -*-
"""
IMF_FBE_selector.py
功能：基于频带熵 (FBE, Frequency Band Entropy) 的 IMF 模态筛选方法。
用于与能量/相关性剪枝方法形成对比实验。
"""

import numpy as np
from scipy.signal import welch, hilbert

def _norm_entropy(p):
    """计算 Shannon 熵并归一化到 [0, 1]"""
    p = np.clip(p, 1e-15, 1.0)
    p = p / p.sum()
    H = -np.sum(p * np.log(p))
    return float(H / np.log(len(p) + 1e-12))

def band_entropy(imf, fs, use_envelope=False, nperseg=None):
    """
    计算 IMF 的频带熵 (FBE)。
    参数
    ----------
    imf : np.ndarray
        单条 IMF 信号。
    fs : float
        采样率。
    use_envelope : bool
        是否在 Hilbert 包络域计算熵。
    nperseg : int
        Welch 分段长度，可不指定。
    返回
    ----------
    fbe : float, [0, 1]
        频带熵，越小越“有结构”，越大越像噪声。
    """
    x = np.asarray(imf, dtype=float)
    if use_envelope:
        x = np.abs(hilbert(x))
    f, Pxx = welch(x, fs=fs, nperseg=nperseg or min(1024, max(256, len(x)//4)))
    Pxx = Pxx + 1e-15
    Pxx /= Pxx.sum()
    return _norm_entropy(Pxx)

def select_imfs_by_FBE(U, fs, use_envelope=True, thr=None, top_k=None):
    """
    基于 FBE 的 IMF 筛选。
    - 可设阈值 thr（保留熵低于该值的 IMF）
    - 或设 top_k（保留熵最低的若干 IMF）
    """
    if U.size == 0:
        return U, np.array([])

    K, N = U.shape
    fbe_vals = np.array([band_entropy(U[k], fs, use_envelope) for k in range(K)])
    print("[FBE] ", " ".join([f"{v:.3f}" for v in fbe_vals]))

    if thr is not None:
        keep = fbe_vals <= thr
    elif top_k is not None:
        keep = np.argsort(fbe_vals) < top_k
        mask = np.zeros(K, dtype=bool)
        mask[keep] = True
        keep = mask
    else:
        # 默认保留一半低熵 IMF
        order = np.argsort(fbe_vals)
        keep = np.zeros(K, dtype=bool)
        keep[order[:K//2]] = True

    U_sel = U[keep]
    print(f"[FBE筛选] kept {U_sel.shape[0]}/{K}, min={fbe_vals.min():.3f}, max={fbe_vals.max():.3f}")
    return U_sel, fbe_vals
