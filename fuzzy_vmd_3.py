
import numpy as np
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from signal_utils import mad_sigma, spectral_entropy # 假设这部分在您的环境中可用

# --- 原始 VMD 辅助函数 (保持不变) ---

rng = np.random.default_rng
_EPS = 1e-12

def estimate_K_by_spectral_peaks(x, fs, K_max=12, min_prom=0.02):
    """通过简单的谱峰计数估计 IMF 数量 K。"""
    X = np.abs(rfft(x))
    if not np.isfinite(X).all():
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    prom = X / (X.max()+1e-12)
    peaks = (prom[1:-1] > prom[:-2]) & (prom[1:-1] > prom[2:]) & (prom[1:-1] > min_prom)
    K = int(np.clip(peaks.sum(), 2, K_max))
    return K

def alpha_from_SE(x, fs, alpha0, beta):
    """根据谱熵(SE)调整带宽正则强度。"""
    alpha = np.clip(alpha0, fs / 50, fs / 5)
    se = spectral_entropy(x, fs)
    return alpha * (1.0 + beta*(se - 0.5))

def soft_threshold(x, thr):
    """L1 软阈值算子：prox_{thr * |.|}(x)"""
    return np.sign(x) * np.maximum(np.abs(x)-thr, 0.0)

def _moving_average_fast(x: np.ndarray, win: int = 5) -> np.ndarray:
    """利用cumsum实现的快速滑动均值 (与您原版功能一致，内联处理)"""
    win = max(1, int(win))
    if win == 1:
        return x
    cumsum = np.cumsum(np.insert(x, 0, 0.0))
    out = (cumsum[win:] - cumsum[:-win]) / float(win)
    pad_left = win // 2
    pad_right = len(x) - len(out) - pad_left
    return np.pad(out, (pad_left, pad_right), mode="edge")

def compute_bandwidth_batch(U):
    """向量化带宽近似计算：所有 IMF 一次性处理 (近似：一阶差分能量)"""
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
                bw = max(100, fs/(4*K))
                low = max(0.5, W[k]-bw/2)/(fs/2)
                high = min(fs/2-1, W[k]+bw/2)/(fs/2)
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
        # W_c = np.copy(W)  # 先拷贝，避免空掩码时出 NaN
        # if np.any(mask):
        #     W_c[mask.ravel()] = (freqs * Uk[mask.ravel()]).sum(axis=1) / ps_sum[mask]
        W_c = np.zeros(K)
        for k in range(K):
            Uk_k = np.abs(rfft(U_norm[k])) ** 2
            ps_sum = Uk_k.sum()
            if ps_sum > 1e-9:
                W_c[k] = np.sum(freqs * Uk_k) / ps_sum
            else:
                W_c[k] = W[k]

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


# --- 模糊逻辑辅助函数 (保持不变) ---

def _tri_mf(x, a, b, c):
    """三角形隶属函数，返回 [0,1]"""
    x = float(x)
    if x <= a or x >= c: return 0.0
    if x == b: return 1.0
    if x < b:
        return (x - a) / (b - a + 1e-12)
    else:
        return (c - x) / (c - b + 1e-12)


def compute_rmse_norm(x, recon):
    """计算归一化均方根误差 (RMSE/Energy)"""
    mse = np.mean((x - recon) ** 2)
    rmse = np.sqrt(mse)
    s_energy = np.sqrt(np.mean(x ** 2)) + 1e-12
    return float(np.clip(rmse / (s_energy + 1e-12), 0.0, 2.0))


def compute_freq_concentration(U, fs):
    """计算频率集中度：(1/平均频率标准差)"""
    K = U.shape[0]
    fc_list = []
    for k in range(K):
        spec = np.abs(rfft(U[k]))
        freqs = rfftfreq(U.shape[1], 1.0 / fs)
        spec_norm = spec / (np.sum(spec) + _EPS)

        # 频率标准差
        mean_f = np.sum(freqs * spec_norm)
        std_f = np.sqrt(np.sum((freqs - mean_f) ** 2 * spec_norm))

        # 集中度 proxy: 标准差的倒数。
        fc = 1.0 / (std_f + _EPS)
        fc_list.append(fc)

    avg_fc = float(np.mean(fc_list))
    fc_norm = np.clip(avg_fc / (fs * 0.05 + 1e-12), 0.0, 1.0)  # 假设 5% 的 fs 是一个合理的 std 范围
    return fc_norm


def compute_energy_overlap(U):
    """计算能量重叠度 (基于能量分布熵的近似)"""
    K = U.shape[0]
    E = np.sum(U ** 2, axis=1) + 1e-12
    p = E / (E.sum() + 1e-12)
    ent = - np.sum(p * np.log(p + 1e-12))
    max_ent = np.log(K + 1e-12)
    overlap = ent / (max_ent + 1e-12)
    return float(np.clip(overlap, 0.0, 1.0))


def fuzzy_inference_simple(rmse_norm, freq_conc, energy_overlap):
    """
    轻量级模糊推理：输入指标，输出 (alpha_scale, k_delta_continuous)
    - alpha_scale: 对 alpha 的缩放因子 (0.6 ~ 1.6)
    - k_delta_continuous: 对 K 的连续调整量 (-1.5 ~ 1.5)
    """
    rm = float(np.clip(rmse_norm, 0.0, 1.0))
    fc = float(np.clip(freq_conc, 0.0, 1.0))  # <-- 使用修正后的 [0, 1] 范围
    eo = float(np.clip(energy_overlap, 0.0, 1.0))

    # --- 隶属度计算 (根据您的原定义) ---
    rm_low = _tri_mf(rm, -0.1, 0.0, 0.3)
    rm_med = _tri_mf(rm, 0.1, 0.4, 0.7)
    rm_high = _tri_mf(rm, 0.5, 1.0, 1.2)

    fc_low = _tri_mf(fc, -0.1, 0.0, 0.3)
    fc_med = _tri_mf(fc, 0.1, 0.4, 0.7)
    fc_high = _tri_mf(fc, 0.5, 1.0, 1.2)

    eo_low = _tri_mf(eo, -0.1, 0.0, 0.3)
    eo_med = _tri_mf(eo, 0.1, 0.4, 0.7)
    eo_high = _tri_mf(eo, 0.5, 1.0, 1.2)

    # --- 模糊规则 (根据您的原定义) ---
    rules = []

    def add_rule(w, alpha_s, k_d):
        if w > 0: rules.append((w, (alpha_s, k_d)))

    add_rule(rm_high * eo_high, 0.8, +1.0)
    add_rule(rm_high * fc_low, 0.75, +1.0)
    add_rule(rm_low * fc_high, 1.25, -1.0)
    add_rule(rm_low * eo_low, 1.1, 0.0)
    add_rule(rm_low * fc_low, 0.9, +1.0)
    add_rule(rm_med * fc_med, 1.0, 0.0)
    add_rule(rm_med * eo_med, 0.95, 0.5)

    # 兜底规则
    add_rule(max(rm_low, rm_med, rm_high) * max(fc_low, fc_med, fc_high) * max(eo_low, eo_med, eo_high) * 0.01,
             1.0, 0.0)

    # --- 解模糊 (质心法/加权平均) ---
    if len(rules) == 0:
        return 1.0, 0.0

    wsum = sum([r[0] for r in rules]) + 1e-12
    alpha_scale = sum([r[0] * r[1][0] for r in rules]) / wsum
    k_delta_c = sum([r[0] * r[1][1] for r in rules]) / wsum

    alpha_scale = float(np.clip(alpha_scale, 0.6, 1.6))
    k_delta_c = float(np.clip(k_delta_c, -1.5, 1.5))

    return alpha_scale, k_delta_c

def fuzzy_adaptive_vmd(
    x, fs,
    K_init=None, alpha_init=None,
    K_min=2, K_max=6, alpha_min=882, alpha_max=8820,
    lam1=0.1, lam2=0.0, lam3=0.0, lam4=0.0,
    max_adapt_steps=5,
    tol_alpha=0.01, tol_K=1,
    iters_per_step=500,
    seed=42,
    use_residual=False,
):
    """
    自适应模糊VMD：
    - x: 输入信号
    - fs: 采样率
    - K_init, alpha_init: 初始值，可由谱峰/谱熵估计
    - lam1-lam4: ADMM参数
    - max_adapt_steps: 最大自适应步骤数
    - tol_alpha/tol_K: 收敛判断阈值
    - iters_per_step: 每步VMD迭代次数
    """
    x = np.asarray(x, dtype=float)
    N = len(x)
    rng = np.random.default_rng(seed)

    # --- 初始化 ---
    if K_init is None:
        K = estimate_K_by_spectral_peaks(x, fs, K_max=K_max)
    else:
        K = int(np.clip(K_init, K_min, K_max))

    if alpha_init is None:
        alpha = fs / 20.0
    else:
        alpha = float(np.clip(alpha_init, alpha_min, alpha_max))

    U_final = None
    W_final = None
    e_final = None

    for step in range(max_adapt_steps):
        # --- 执行单步 VMD ---
        U, W, e = admm_ul_rmsa_vmd(
            x, fs, K, alpha, lam1, lam2, lam3, lam4,
            iters=iters_per_step,
            seed=seed,
            use_residual=use_residual
        )

        recon = np.sum(U, axis=0) + (e if use_residual else 0.0)

        # --- 评估指标 ---
        rmse_norm = compute_rmse_norm(x, recon)
        freq_conc = compute_freq_concentration(U, fs)
        energy_overlap = compute_energy_overlap(U)

        # --- 模糊推理 ---
        alpha_scale, k_delta = fuzzy_inference_simple(rmse_norm, freq_conc, energy_overlap)

        # --- 更新参数 ---
        alpha_new = np.clip(alpha * alpha_scale, alpha_min, alpha_max)
        K_new = int(np.clip(round(K + k_delta), K_min, K_max))

        # --- 打印调试信息 ---
        print(f"[Step {step+1}] K={K}->{K_new}, alpha={alpha:.3f}->{alpha_new:.3f}, RMSE={rmse_norm:.3f}, FreqConc={freq_conc:.3f}, Overlap={energy_overlap:.3f}")

        # --- 稳定性检查 ---
        delta_alpha = abs(alpha_new - alpha) / (alpha + 1e-12)
        delta_K = abs(K_new - K)
        if delta_alpha < tol_alpha and delta_K <= tol_K:
            print(f"自适应提前收敛: delta_alpha={delta_alpha:.4f}, delta_K={delta_K}")
            U_final, W_final, e_final = U, W, e
            break

        # --- 准备下一步 ---
        alpha = alpha_new
        K = K_new
        U_final, W_final, e_final = U, W, e

    return U_final, W_final, e_final, K, alpha

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
def plot_decomposition(x, fs, U, W, e, title="自适应VMD分解结果"):
    """可视化分解结果，重点展示能量分配和频率差异"""
    N = len(x)
    t = np.arange(N) / fs
    K = U.shape[0]

    fig = plt.figure(figsize=(12, 3 + 2 * K))
    gs = GridSpec(K + 3, 1, figure=fig, hspace=0.5)

    # 原始信号
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(t, x)
    ax0.set_title("signal")
    ax0.set_xlabel("time(s)")
    ax0.set_ylabel("zhenfu")

    # 各IMF分量（标注频率）
    for k in range(K):
        ax = fig.add_subplot(gs[k + 1])
        ax.plot(t, U[k])
        ax.set_title(f"IMF {k + 1} (fs: {W[k]:.1f} Hz, energy: {np.sum(U[k] ** 2):.3e})")
        ax.set_xlabel("time(s)")
        ax.set_ylabel("ZHENFU")

    # 残差
    ax_res = fig.add_subplot(gs[K + 1])
    ax_res.plot(t, e)
    ax_res.set_title(f"E (energy: {np.sum(e ** 2) / np.sum(x ** 2):.2%})")
    ax_res.set_xlabel("time(s)")
    ax_res.set_ylabel("Zhenfu")

    plt.tight_layout()
    return fig
if __name__ == "__main__":
    fs = 44100
    t = np.arange(0, 0.5, 1/fs)
    x = (0.5*np.sin(2*np.pi*500*t) +
         0.3*np.sin(2*np.pi*1500*t) +
         0.2*np.sin(2*np.pi*3000*t) +
         0.1*np.random.randn(len(t)))

    U, W, e, K, alpha = fuzzy_adaptive_vmd(x, fs, use_residual=True)

    print("分解 IMF 数量:", U.shape[0])
    print("分解 IMF:", U)
    print("中心频率:", W)
    print("残差 e 能量:", np.sum(e ** 2))
    fig = plot_decomposition(x, fs, U, W, e)
    plt.show()
