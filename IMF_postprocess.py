# modal_postprocess.py
# 模态增强与噪声抑制（IMF 剪枝/合并、加权与重构）
from __future__ import annotations

import numpy as np
import pywt
from signal_utils import snr_estimate


def prune_merge_imfs(U: np.ndarray, energy_thr: float = 0.01, corr_thr: float = 0.9) -> np.ndarray:
    """
    基于能量与相关性的 IMF 剪枝与合并。

    参数
    ----------
    U : np.ndarray, shape = (K, N)
        输入 IMF 组（K 条 IMF，每条长度 N）。
    energy_thr : float
        能量占比阈值。低于该占比的 IMF 将被丢弃。
    corr_thr : float
        相邻 IMF 合并的相关系数阈值（建议理解为“绝对相关阈值”）。

    返回
    ----------
    merged_IMFs : np.ndarray, shape = (K', N)
        处理后的 IMF 组（K' ≤ K）。

    说明
    ----------
    - 先按 IMF 能量占比筛选（避免弱 IMF 干扰）。
    - 再检测相邻 IMF 的（绝对）相关系数，若高于阈值则合并（相当于带宽略放宽的单模态）。
    - 对于极端设置（把 IMF 全筛掉），做“保底”处理：保留能量最大的 1 条。
    """
    # 如果输入为空，直接返回
    if U.size == 0:
        return U

    # K: IMF 数；N: 每条 IMF 的长度
    K, N = U.shape

    # === 能量占比筛选 ===
    energies = np.sum(U ** 2, axis=1)  # 各 IMF 能量
    total_energy = float(np.sum(energies)) + 1e-12
    energies_ratio = energies / total_energy  # 能量占比
    print("  [IMF能量占比]", ["{:.4f}".format(e) for e in energies_ratio])
    keep_mask = energies_ratio >= energy_thr

    # 先保留满足能量阈值的 IMF
    U2 = U[keep_mask]

    # 若全部被筛掉：保底保留能量最大的 IMF（避免后续空数组报错）
    if U2.shape[0] == 0:
        max_idx = int(np.argmax(energies))
        U2 = U[max_idx:max_idx + 1]

    # === 合并相邻高相关 IMF ===
    merged = []
    i = 0
    while i < len(U2):
        cur = U2[i]
        if i < len(U2) - 1:
            nxt = U2[i + 1]
            # 相关系数；当序列几乎常量时可能出现 nan，这里用 0 代替
            r = np.corrcoef(cur, nxt)[0, 1]
            if not np.isfinite(r): r = 0.0
            if abs(r) > corr_thr:
                cur = cur + nxt
                i += 1  # 跳过 nxt
        merged.append(cur)
        i += 1

    U_merged = np.vstack(merged)
    print(f"  [剪枝阈值={energy_thr}, corr_thr={corr_thr}] 最终保留 {U_merged.shape[0]}/{K} 个 IMF")
    return U_merged


def wavelet_denoise_imf(u: np.ndarray, wavelet: str = "db4", level: int = 3) -> np.ndarray:
    """
    对单条 IMF 做小波软阈值降噪（VisuShrink）。

    参数
    ----------
    u : np.ndarray, shape = (N,)
        单条 IMF。
    wavelet : str
        小波基名，例如 "db4"、"sym8" 等。
    level : int
        期望分解层数（将自动裁剪到允许的最大层数）。

    返回
    ----------
    u_den : np.ndarray, shape = (N,)
        降噪后的 IMF（长度与输入对齐）。

    实现细节
    ----------
    - MAD 估计噪声：sigma = median(|detail|) / 0.6745；采用最高层细节系数估计。
    - VisuShrink 阈值：thr = sigma * sqrt(2 * ln(N))。
    - 对 level 做边界检查：若序列过短导致 max_level < 1，则直接返回原信号。
    """
    n = len(u)
    if n == 0:
        return u

    # 自动裁剪 level，避免 “Level value too high” 异常
    wobj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(n, wobj.dec_len)
    level = int(min(level, max_level))
    if level < 1:
        # 无法做有效分解，直接返回原样
        return u.copy()

    # 多层小波分解（coeffs = [cA_L, cD_L, cD_{L-1}, ..., cD_1]）
    coeffs = pywt.wavedec(u, wavelet, level=level)

    # 用最高层细节系数估计噪声方差的 sigma（稳健）
    # 注：也可以选用最细层 cD1；经验上两者都常见，这里沿用你原本“末层细节”做法
    cD = coeffs[-1]
    sigma = np.median(np.abs(cD)) / 0.6745
    sigma = float(max(sigma, 1e-12))  # 数值下限，避免 0

    # VisuShrink 通用阈值
    thr = sigma * np.sqrt(2.0 * np.log(max(n, 2)))

    # 对每一层细节系数做软阈值，近似保留低频逼近系数
    den_coeffs = [coeffs[0]] + [pywt.threshold(c, thr, mode="soft") for c in coeffs[1:]]

    # 重构并裁剪到原长度（因为边界扩展可能多出 1~几样本）
    u_den = pywt.waverec(den_coeffs, wavelet)[:n]
    return u_den.astype(u.dtype, copy=False)


def noise_template_similarity(u: np.ndarray, template: np.ndarray | None) -> float:
    """
    与“噪声模板”的反相似度：1 - cos_sim(u, template)
    - 越小：越像模板（更像噪声）
    - 越大：越不像模板（更像有效信号）

    当 template 为 None 或空时，返回 0（表示“无区分度”）。
    """
    if template is None or len(template) == 0:
        return 0.0

    n = min(len(u), len(template))
    a = u[:n]
    b = template[:n]
    na = float(np.linalg.norm(a)) + 1e-12
    nb = float(np.linalg.norm(b)) + 1e-12
    cos_sim = float(np.dot(a, b) / (na * nb))
    # 反相似度（距离）：1 - cos_sim
    return 1.0 - cos_sim


def compute_weights(U: np.ndarray, e: np.ndarray,
                    eta: tuple[float, float, float] = (0.5, 0.3, 0.2),
                    noise_template: np.ndarray | None = None) -> np.ndarray:
    """
    计算各 IMF 的融合权重 w（softmax 归一化），用于后续重构。

    指标与权重
    ----------
    - p_t：能量占比（越大越重要）
    - s_t：SNR（越大越重要）
    - r_t：与噪声模板的“反相似度”（越大越不像噪声 ⇒ 越重要）
    - score = eta[0]*p_t + eta[1]*s_t + eta[2]*r_t
    - w = softmax(score)

    参数
    ----------
    U : np.ndarray, shape = (K, N)
        IMF 组。
    e : np.ndarray
        残差信号（本函数不直接使用，但保留参数以兼容上游接口）。
    eta : tuple of 3 floats
        三个指标的线性权重，建议和为 1（函数内部不强制，但会做数值稳定）。
    noise_template : np.ndarray | None
        外部先验的“噪声模板”。

    返回
    ----------
    w : np.ndarray, shape = (K,)
        归一化的融合权重，非负且和为 1；若 K==0 返回空数组。
    """
    K = U.shape[0]
    if K == 0:
        return np.zeros((0,), dtype=float)

    # --- 能量占比 ---
    ek = np.sum(U ** 2, axis=1)
    p = ek / (np.sum(ek) + 1e-12)

    # --- SNR 估计（snr_estimate 建议返回线性或 dB 皆可；此处做 min-max 归一化）---
    snrs = np.array([snr_estimate(U[k]) for k in range(K)], dtype=float)

    # --- 反相似度（与噪声模板）---
    rs = np.array([noise_template_similarity(U[k], noise_template) for k in range(K)], dtype=float)

    # --- 归一化到 [0,1] 的小工具 ---
    def norm01(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        vmax, vmin = float(np.max(v)), float(np.min(v))
        if vmax - vmin < 1e-12:
            # 常量向量：返回 0.5，表示“有信息但不可区分”
            return np.ones_like(v) * 0.5
        return (v - vmin) / (vmax - vmin)

    p_t = p / (np.sum(p) + 1e-12)     # 再次约束（即使 p 已经是占比）
    s_t = norm01(snrs)
    r_t = norm01(rs)

    # --- 线性融合得分 ---
    eta = np.asarray(eta, dtype=float)
    if not np.isfinite(eta).all() or eta.shape != (3,):
        eta = np.array((0.5, 0.3, 0.2), dtype=float)
    # 允许 eta 不和为 1：做一个柔性归一化
    eta_sum = float(np.sum(eta)) + 1e-12
    eta = eta / eta_sum

    score = eta[0] * p_t + eta[1] * s_t + eta[2] * r_t

    # --- softmax（数值稳定）---
    score = np.asarray(score, dtype=float)
    ex = np.exp(score - float(np.max(score)))
    w = ex / (float(np.sum(ex)) + 1e-12)
    return w


def reconstruct(U: np.ndarray, e: np.ndarray, weights: np.ndarray,
                wavelet: str = "db4", level: int = 3, use_wavelet: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    对 IMF 逐条小波降噪后，依据权重加权融合，并与残差 e 相加，得到去噪重构信号。

    参数
    ----------
    U : np.ndarray, shape = (K, N)
        IMF 组。
    e : np.ndarray, shape = (N,)
        残差/噪声项（来自分解器）。
    weights : np.ndarray, shape = (K,)
        各 IMF 融合权重（通常来自 compute_weights）。
    wavelet : str
        小波基名。
    level : int
        小波分解层数（自动裁剪到允许上限）。

    返回
    ----------
    s_hat : np.ndarray, shape = (N,)
        重构后的时域信号。
    denoised : np.ndarray, shape = (K, N)
        每条 IMF 的降噪版本（与 U 对齐；若 K==0 返回 shape=(0, N)）。
    """
    K, N = U.shape[0], (U.shape[1] if U.ndim == 2 and U.shape[0] > 0 else len(e))

    if K == 0:
        # 没有 IMF：直接把残差作为重构结果；返回空的 denoised
        s_hat = e.astype(float, copy=True)
        denoised = np.zeros((0, N), dtype=s_hat.dtype)
        return s_hat, denoised

    # 对每条 IMF 做小波软阈降噪（长度保持与原 IMF 一致）
    if use_wavelet:
        # 对每条 IMF 做小波软阈降噪
        denoised_list = [wavelet_denoise_imf(U[k], wavelet, level) for k in range(K)]
    else:
        # 不做小波处理，直接使用 IMF
        denoised_list = [U[k] for k in range(K)]
    denoised = np.vstack(denoised_list)

    # 保障权重形状与和为 1（即使外部传入稍有偏差）
    w = np.asarray(weights, dtype=float).reshape(K)
    w = np.maximum(w, 0.0)
    ws = float(np.sum(w)) + 1e-12
    w = w / ws

    # 融合各 IMF，并叠加残差 e
    s_hat = np.sum(w[:, None] * denoised, axis=0) + e
    print(f"  [重构前] IMF数={K}, mean_SNR={np.mean([snr_estimate(u) for u in U]):.2f} dB")
    print(f"  [重构后] SNR={snr_estimate(s_hat):.2f} dB")
    return s_hat.astype(float, copy=False), denoised
