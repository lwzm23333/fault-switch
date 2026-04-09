# ul_rmsa_vmd.py
# 执行VMD分解

import numpy as np
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from signal_utils import mad_sigma, spectral_entropy

rng = np.random.default_rng
_EPS = 1e-12

def estimate_K_by_spectral_peaks(x, fs, K_max=12, min_prom=0.02):
    # 简单谱峰计数（可替换更高级谱聚类）"""
    #     通过简单的谱峰计数估计 IMF 数量 K。
    #     - 使用归一化谱幅的“本地极大”计数；最少 2、最多 K_max。
    #     - 注意：这里不考虑峰宽/显著性，可替换为 scipy.signal.find_peaks 的 prominence 实现。
    #     """
    X = np.abs(rfft(x))         # 实数 FFT 的幅值谱（非负频率半谱）
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prom = X / (X.max()+1e-12)      # 归一化到[0,1]，避免除零
    # 在内部点[1:-1]找局部极大：prom[i] > prom[i-1] 且 prom[i] > prom[i+1]，并且大于阈值
    peaks = (prom[1:-1] > prom[:-2]) & (prom[1:-1] > prom[2:]) & (prom[1:-1] > min_prom)
    # 计数并裁剪到[2, K_max]
    K = int(np.clip(peaks.sum(), 2, K_max))
    return K

def alpha_from_SE(x, fs, alpha0, beta):
    """
    根据谱熵(SE)调整带宽正则强度：
    alpha = alpha0 * (1 + beta * (SE - 0.5))
    SE ∈ [0,1] 越接近 1 表明频谱越均匀/越“白”，适当增加带宽惩罚。
    """
    alpha = np.clip(alpha0, fs / 50, fs / 5)
    se = spectral_entropy(x, fs)       # 计算谱熵值
    return alpha * (1.0 + beta*(se - 0.5))    # 调整宽带正则强度

def soft_threshold(x, thr):
    """L1 软阈值算子：prox_{thr * |.|}(x)"""
    return np.sign(x) * np.maximum(np.abs(x)-thr, 0.0)


def _moving_average_fast(x: np.ndarray, win: int = 5) -> np.ndarray:
    """利用cumsum实现的快速滑动均值"""
    win = max(1, int(win))
    if win == 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = (cumsum[win:] - cumsum[:-win]) / float(win)
    # 补边缘（保持 same 模式）
    pad_left = win // 2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode="edge")


def soft_threshold(x, thr):
    return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)


def compute_bandwidth_batch(U):
    """向量化带宽近似计算：所有 IMF 一次性处理"""
    d = np.diff(U, axis=1, prepend=U[:, :1])
    return float(np.sum(d * d))


def admm_ul_rmsa_vmd(
    x, fs, K, alpha, lam1, lam2, lam3, lam4,
    iters=200, tol=1e-6, seed=42,
    use_residual=False,        # 是否启用稀疏残差 e
    use_freq_proj="hilbert",      # 是否对 IMF 做频域带通投影（可先 False，训练稳定后再打开）  /"fft_mask"/hilbert
):
    """
    干净版 UL-RMSA-VMD (ADMM)
    - IMF 更新：残差投影 -> 平滑 -> 最小二乘增益  (rho=1.0)
    - 残差 e：MAD 自适应软阈值 + 去直流/高通 + 动态能量 cap
    - 统一的对偶变量更新 z = z + mu * (r - e)
    - 频率批量更新：用单位能量归一的 IMF 估计中心频率
    """
    import numpy as np
    from numpy.fft import rfft, irfft, rfftfreq
    from scipy.signal import fftconvolve

    np.random.seed(seed)
    x = np.asarray(x, dtype=float)
    N = x.size

    # --- 初始化 ---
    U = np.zeros((K, N), dtype=float)
    W = np.linspace(0.05 * fs, 0.45 * fs, K).astype(float)  # 初始中心频
    e = np.zeros(N, dtype=float)
    z = np.zeros(N, dtype=float)
    mu = 1.0
    X = x.copy()
    last_obj = np.inf

    # 常量
    t = np.arange(N, dtype=float) / fs
    freqs = rfftfreq(N, 1.0 / fs)
    _EPS = 1e-12

    # 根据 alpha 生成平滑窗口长度（经验）
    base_win = int(np.clip(round(3 + 10 * float(alpha)), 3, 51))
    if base_win % 2 == 0:
        base_win += 1

    # 简单快速移动平均（假设你已有同名实现；若无，可内联实现）
    def _moving_average_fast(x, win):
        if win <= 1:
            return x
        c = np.convolve(x, np.ones(win) / float(win), mode="same")
        return c

    # IMF 组带宽（假设你已有实现；若无，这里给个稳妥近似）
    def compute_bandwidth_batch(U):
        # 近似：每个 IMF 的一阶差分能量作为带宽 proxy
        d = np.diff(U, axis=1)
        return float(np.sum(d * d))

    # 软阈值
    def soft_threshold(x, thr):
        return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)

    for it in range(iters):
        # ----- 主残差 y -----
        y = X - (e if use_residual else 0.0) + z / mu
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        rem = y.copy()

        # ===== 更新每个 IMF =====
        for k in range(K):
            # 频率裁剪，防止数值失真
            if not np.isfinite(W[k]):
                W[k] = 0.25 * fs
            W[k] = float(np.clip(W[k], 1.0, 0.49 * fs))

            # 基载波（简化版调频基）
            carrier = np.cos(2.0 * np.pi * W[k] * t)
            carrier = np.nan_to_num(carrier, nan=0.0)

            # 残差在载波上的投影（FFT 卷积）
            proj = fftconvolve(rem, carrier[::-1], mode="same")
            proj = proj / (np.linalg.norm(carrier) + _EPS)

            # 平滑
            u = 0.5 * proj + 0.25 * np.roll(proj, 1) + 0.25 * np.roll(proj, -1)
            u = _moving_average_fast(u, win=base_win)

            # 最小二乘增益 (rho=1.0，更“敢吸能量”)
            denom = float(np.sum(u * u)) + _EPS
            g = float(np.dot(rem, u) / denom)
            u = g * u

            if use_freq_proj == "fft_mask":
                U_f = rfft(rem)
                bw = fs / (2 * K)  # 每个 IMF 平均带宽
                mask = (np.abs(freqs - W[k]) < bw / 2).astype(float)
                U_f *= mask
                u = irfft(U_f, n=N)

            elif use_freq_proj == "hilbert":
                from scipy.signal import hilbert, butter, filtfilt
                # Hilbert 解析信号
                analytic = hilbert(rem)
                # 带通滤波器设计（中心 W[k], 带宽 fs/(2K)）
                bw = fs / (2 * K)
                low = max(1.0, W[k] - bw / 2) / (fs / 2)
                high = min(0.49 * fs, W[k] + bw / 2) / (fs / 2)
                if high <= low:
                    u = np.real(analytic)  # fallback
                else:
                    b, a = butter(3, [low, high], btype="band")
                    u = filtfilt(b, a, np.real(analytic))

            U[k] = u
            rem -= u

        # ===== 更新稀疏残差 e =====
        recon = np.sum(U, axis=0)
        r = X - recon

        if use_residual:
            # 自适应 MAD 阈值（≈3σ）
            mad = np.median(np.abs(r - np.median(r))) + _EPS
            tau = 4.0 * 1.4826 * mad
            e = soft_threshold(r + z / mu, tau)

            # 去直流 + 高通平滑
            e = e - np.mean(e)
            hp_win = min(257, max(31, int(N // 48)))
            e = e - _moving_average_fast(e, win=hp_win)

            # 动态能量 cap：保证 IMF 至少吸收一定能量
            E_seg = np.sum(X * X) + _EPS
            E_u = np.sum(recon * recon)
            # 留 10% 缓冲，cap ∈ [0.1, 0.5]
            cap = min(0.5, max(0.05, 1.0 - E_u / E_seg - 0.1))
            E_e = np.sum(e * e)
            if E_e / E_seg > cap:
                e *= np.sqrt(cap / (E_e / E_seg + _EPS))
        else:
            e[:] = 0.0

        # ===== 批量更新中心频 =====
        denom = np.sqrt(np.mean(U * U, axis=1, keepdims=True)) + _EPS
        U_norm = U / denom
        Uk = np.abs(rfft(U_norm, axis=1)) ** 2  # (K, N//2+1)

        ps_sum = Uk.sum(axis=1, keepdims=True)
        mask = ps_sum > 1e-9
        W_c = np.copy(W)  # 先拷贝，避免空掩码时出 NaN
        if np.any(mask):
            W_c[mask.ravel()] = (freqs * Uk[mask.ravel()]).sum(axis=1) / ps_sum[mask]

        # 频率平滑（lam3 越大 -> 更新越慢）
        s = 0.8 + 0.19 * (1.0 / (1.0 + lam3))
        W = s * W + (1.0 - s) * W_c
        W = np.clip(W, 1.0, 0.49 * fs)

        # ===== 对偶变量更新（统一写法）=====
        z = z + mu * (r - e)

        # ===== 目标函数 & 收敛判据 =====
        bw = compute_bandwidth_batch(U)
        obj = float(np.sum((X - recon - e) ** 2) + lam1 * np.sum(np.abs(e)) + lam2 * bw)

        if not np.isfinite(obj):
            U = np.nan_to_num(U)
            e = np.nan_to_num(e)
            W = np.nan_to_num(W, nan=0.25 * fs)
            break

        if abs(last_obj - obj) < tol:
            break
        last_obj = obj

    return U, W, e

def cross_segment_frequency_lock(W_list, n_clusters):
    """
    跨段频率锁定：将各段得到的 {W_k} 聚类，输出有序的聚类中心，作为全局对齐频带。
    输入：
      - W_list: List[np.ndarray]，每段的频率序列（长度可不同）
      - n_clusters: 若为空，用各段长度的中位数作为聚类数的初始估计
    返回：
      - centers: (C,) 已排序的聚类中心（Hz）

    假设：
      - W 值均为 Hz，且处于 (0, fs/2) 范围内（外部保证或事先裁剪）
    """
    # 过滤空或 None
    seqs = [np.asarray(w, dtype=float).ravel() for w in W_list if w is not None and len(w) > 0]
    if len(seqs) == 0:
        return np.array([], dtype=float)

    all_w = np.concatenate(seqs)            # 拼接所有段的频率
    all_w = all_w[np.isfinite(all_w)]       # 仅保留有限值
    if all_w.size == 0:
        return np.array([], dtype=float)

    if n_clusters is None:
        # 用各段长度的中位数作为初估
        med = int(np.median([len(s) for s in seqs]))
        n_clusters = max(2, min(med, all_w.size))  # 防止聚类数超过样本数
    else:
        n_clusters = int(np.clip(n_clusters, 1, all_w.size))
        if n_clusters < 2:
            # 至少 2 类才能形成“带”概念；若确实为 1，则直接返回均值
            return np.array([float(np.mean(all_w))], dtype=float)

    # 进行 KMeans 聚类（n_init=10 提高稳定性，random_state=0 保证可复现）
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    km.fit(all_w.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.ravel().astype(float))
    return centers
