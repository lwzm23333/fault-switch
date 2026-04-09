# === BEGIN ADDITION: fuzzy utils for adaptive VMD ===
import numpy as np

def _tri_mf(x, a, b, c):
    """三角形隶属函数，返回 [0,1]"""
    # a <= b <= c is expected
    x = float(x)
    if x <= a or x >= c:
        return 0.0
    if x == b:
        return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    else:
        return (c - x) / (c - b + 1e-12)

def normalize_to01(v, vmin, vmax):
    """简单线性归一化并 clip"""
    if vmax <= vmin:
        return 0.0
    return float(np.clip((v - vmin) / (vmax - vmin), 0.0, 1.0))

def compute_rmse_norm(x, recon):
    mse = np.mean((x - recon) ** 2)
    rmse = np.sqrt(mse)
    # 归一化策略：可以基于信号能量做缩放
    s_energy = np.sqrt(np.mean(x**2)) + 1e-12
    return float(np.clip(rmse / (s_energy + 1e-12), 0.0, 2.0))  # 可能超 1，后续可clip/map

from numpy.fft import fft, ifft
_EPS = 1e-12
def compute_freq_concentration(U, fs):
    """
    计算频率集中度：对每个 IMF 用其功率谱，freq_conc = max_bin_power / sum(bin_power)
    最后取 IMF 的平均（越接近 1 越集中）
    """
    from numpy.fft import rfft
    K = U.shape[0]
    N = U.shape[1]
    freqs = None
    fc_list = []
    for k in range(K):
        spec = np.abs(fft(U[k]))[:U.shape[1] // 2]
        spec /= np.sum(spec) + _EPS
        freqs = np.arange(len(spec)) / len(spec) * fs / 2
        mean_f = np.sum(freqs * spec)
        std_f = np.sqrt(np.sum((freqs - mean_f) ** 2 * spec))
        fc_list.append(1.0 / (std_f + _EPS))
    return float(np.mean(fc_list))

def compute_energy_overlap(U):
    """
    能量重叠度的近似：在时间域上计算归一化能量分布，并用 min(prev,next) 衡量重合比例。
    更鲁棒的方法是计算各 IMF 频谱上的重合面积比（此处做简单时间域 proxy）。
    """
    K = U.shape[0]
    E = np.sum(U**2, axis=1) + 1e-12
    p = E / (E.sum() + 1e-12)  # 每个 IMF 所占能量比例
    # overlap proxy: 如果许多 IMF 的能量都接近平均，说明更重叠（取熵反比）
    ent = - np.sum(p * np.log(p + 1e-12))
    max_ent = np.log(K + 1e-12)
    overlap = ent / (max_ent + 1e-12)  # ∈ [0,1], 越大代表越“重叠/混叠”
    return float(np.clip(overlap, 0.0, 1.0))


def fuzzy_inference_simple(rmse_norm, freq_conc, energy_overlap):
    """
    你之前给的轻量级模糊推理的实现，包含在此（做了微小整理）
    返回 (alpha_scale, k_delta_continuous)
    """
    rm = float(np.clip(rmse_norm, 0.0, 2.0))
    if rm > 1.0:
        rm = 1.0
    fc = float(np.clip(freq_conc, 0.0, 1.0))
    eo = float(np.clip(energy_overlap, 0.0, 1.0))

    rm_low = _tri_mf(rm, -0.1, 0.0, 0.3)
    rm_med = _tri_mf(rm, 0.1, 0.4, 0.7)
    rm_high = _tri_mf(rm, 0.5, 1.0, 1.2)

    fc_low = _tri_mf(fc, -0.1, 0.0, 0.3)
    fc_med = _tri_mf(fc, 0.1, 0.4, 0.7)
    fc_high = _tri_mf(fc, 0.5, 1.0, 1.2)

    eo_low = _tri_mf(eo, -0.1, 0.0, 0.3)
    eo_med = _tri_mf(eo, 0.1, 0.4, 0.7)
    eo_high = _tri_mf(eo, 0.5, 1.0, 1.2)

    rules = []

    def add_rule(w, alpha_s, k_d):
        if w > 0:
            rules.append((w, (alpha_s, k_d)))

    add_rule(rm_high * eo_high, 0.8, +1.0)
    add_rule(rm_high * fc_low, 0.75, +1.0)
    add_rule(rm_low * fc_high, 1.25, -1.0)
    add_rule(rm_low * eo_low, 1.1, 0.0)
    add_rule(rm_med * fc_med, 1.0, 0.0)
    add_rule(rm_med * eo_med, 0.95, 0.5)
    add_rule(max(rm_low, rm_med, rm_high) * max(fc_low, fc_med, fc_high) * max(eo_low, eo_med, eo_high),
             1.0, 0.0)

    if len(rules) == 0:
        return 1.0, 0.0
    wsum = sum([r[0] for r in rules]) + 1e-12
    alpha_scale = sum([r[0] * r[1][0] for r in rules]) / wsum
    k_delta_c = sum([r[0] * r[1][1] for r in rules]) / wsum
    alpha_scale = float(np.clip(alpha_scale, 0.6, 1.6))
    k_delta_c = float(np.clip(k_delta_c, -1.5, 1.5))
    return alpha_scale, k_delta_c
# === END ADDITION ===
