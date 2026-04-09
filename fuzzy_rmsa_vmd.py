# ul_rmsa_vmd.py
# 执行VMD分解
import scipy.signal as sps
import numpy as np
from numpy.fft import fft, ifft, rfft, irfft, rfftfreq
from scipy.signal import fftconvolve
from sklearn.cluster import KMeans
from signal_utils import mad_sigma, spectral_entropy
from ul_rmsa_vmd import fuzzy_inference_simple, compute_energy_overlap, compute_freq_concentration, compute_rmse_norm
import numpy as np
from numpy.fft import rfft, irfft, rfftfreq
from scipy.signal import fftconvolve
from scipy.signal import hilbert

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
    se = spectral_entropy(x, fs)
    if se > 1:
        se = se / np.log(len(x))  # 归一化到 [0,1]
    alpha = alpha0 * (1.0 + beta * (se - 0.5))
    alpha = np.clip(alpha, fs / 50, fs / 5)
    return alpha, se  # 调整宽带正则强度

def soft_threshold(x, thr):
    """L1 软阈值算子：prox_{thr * |.|}(x)"""
    return np.sign(x) * np.maximum(np.abs(x)-thr, 0.0)

def mad(x):
    return np.median(np.abs(x - np.median(x))) * 1.4826

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
        x, fs, K, alpha, lam1, lam2=0, lam3=0, lam4=0,
        iters=200, tol=1e-5, seed=42,
        use_residual=True,
        use_freq_proj="hilbert",
        fuzzy_enable=True,
        fuzzy_interval=15,
        fuzzy_allow_K_change=True,
        record_alpha_K=True,
        debug=True
):
    """
    改进版自适应VMD分解（结合逐模剥离和低通滤波）
    解决IMF同质化、能量分配失衡问题
    """

    # ---------- 辅助函数 ----------
    def mad(x):
        return np.median(np.abs(x - np.median(x))) + 1e-12

    def soft_threshold(x, thr):
        return np.sign(x) * np.maximum(np.abs(x) - thr, 0.0)

    def fuzzy_inference(rmse_norm, freq_conc, energy_overlap):
        """模糊推理：根据分解质量调整参数"""
        a_scale = 1.0 + 0.3 * (1.0 - rmse_norm)  # alpha调整因子
        if freq_conc < 0.3:
            k_delta = 1  # 频率分散→增加K
        elif freq_conc > 0.7:
            k_delta = -1  # 频率集中→减少K
        else:
            k_delta = 0
        return np.clip(a_scale, 0.7, 1.3), k_delta

    # ---------- 初始化 ----------
    np.random.seed(seed)
    x = np.asarray(x, dtype=float)
    N = len(x)
    t = np.arange(N) / fs  # 时间轴
    freqs = rfftfreq(N, 1.0 / fs)  # 频率轴（loop外预计算）
    _EPS = 1e-12
    rng = np.random.default_rng(seed)

    # 初始化IMF矩阵（K行N列），赋予初始能量避免全零
    U = rng.normal(0, 0.1, (K, N))  # 小幅度随机初始化
    # 中心频率初始化为均匀分布
    W = np.linspace(0.05 * fs, 0.45 * fs, K)
    e = np.zeros(N)  # 稀疏残差
    z = np.zeros(N)  # ADMM对偶变量
    mu = 5.0  # 惩罚参数
    X = x.copy()

    # 自适应参数
    alpha_curr = alpha
    K_curr = K
    alpha_list, K_list, rmse_list = [], [], []
    last_fuzzy_iter = -999

    # ---------- 主循环 ----------
    for it in range(iters):
        # 计算ADMM残差（原始信号 - 稀疏残差 + 对偶项）
        resid = X - e + z / (mu + _EPS)

        # 逐模剥离残差（核心改进：每次从剩余残差中提取一个IMF）
        rem = resid.copy()  # 剩余残差，初始为总残差
        for k in range(K_curr):
            # 1. 计算当前剩余残差的解析信号
            analytic = sps.hilbert(rem)
            analytic_norm = analytic / (np.max(np.abs(analytic)) + _EPS)  # 归一化

            # 2. 解调至基带（根据当前中心频率）
            exponent = np.exp(-2j * np.pi * W[k] * t)
            demod = analytic_norm * exponent  # 解调后的复信号

            # 3. 低通滤波提取基带信号（关键：分离不同频率分量）
            # 带宽自适应：根据K和采样率动态调整
            bw = max(50.0, fs / (2 * K_curr))  # 最小50Hz，或按K分配
            cutoff = min(bw / (fs / 2), 0.99)  # 归一化截止频率（<1）

            if cutoff <= 0:
                base = np.real(demod)
            else:
                # 3阶巴特沃斯低通滤波
                b, a = sps.butter(3, cutoff, btype='low')
                base = sps.filtfilt(b, a, np.real(demod))  # 零相位滤波

            # 4. 最小二乘增益计算（让IMF最佳逼近当前剩余残差）
            rem_before = rem.copy()  # 记录当前残差用于增益计算
            denom = np.sum(base * base) + _EPS
            g = np.dot(rem_before, base) / denom  # 最优增益
            g = np.clip(g, -1e3, 1e3)  # 限制增益避免极端值

            # 5. 生成IMF分量并更新剩余残差
            u = g * base  # 带增益的IMF分量
            rem = rem - u  # 从剩余残差中剥离当前IMF

            # 6. 存储IMF（确保数值稳定性）
            if not np.all(np.isfinite(u)) or np.max(np.abs(u)) > 1e6:
                if debug:
                    print(f"[WARN] it={it}, IMF{k} 异常，重置")
                u = np.zeros(N)
            U[k] = u  # 存储更新后的IMF

            # 调试输出
            if debug and (it % 50 == 0):
                print(f"[it={it}, IMF{k}] 增益={g:.3e}, 残差能量={np.linalg.norm(rem):.3e}")

        # 7. 更新稀疏残差e（L1正则化）
        total_recon = np.sum(U, axis=0)  # 所有IMF的重构信号
        r = X - total_recon - z / mu  # ADMM原始残差
        r -= np.mean(r)  # 去除直流分量

        # 自适应阈值
        sigma = mad(r)
        thr = lam1 * sigma
        e_new = soft_threshold(r, thr)

        # 收紧残差能量约束（强制IMF承担主要能量）
        energy_r = np.linalg.norm(r)
        energy_e = np.linalg.norm(e_new)
        if energy_e > 0.2 * energy_r:  # 残差能量不超过20%
            e_new *= 0.2 * energy_r / (energy_e + _EPS)
        e = e_new

        # 8. 更新对偶变量z
        z += mu * (X - total_recon - e)

        # 9. 更新中心频率（带功率谱平滑和跳变限制）
        for k in range(K_curr):
            # 计算功率谱
            Uk = rfft(U[k])
            Uk_power = np.abs(Uk) ** 2  # 用功率谱计算频率中心
            ps_sum = np.sum(Uk_power)

            if ps_sum > 1e-12:  # 能量足够时更新频率
                # 频率中心 = 功率谱加权平均
                cent = np.sum(freqs * Uk_power) / ps_sum

                # 限制频率跳变幅度（避免突变）
                max_shift = 0.1 * fs / K_curr  # 最大移动量：带宽的0.1倍
                W[k] = np.clip(cent, W[k] - max_shift, W[k] + max_shift)

                # 频率范围约束
                W[k] = np.clip(W[k], 0.01 * fs, 0.49 * fs)
            # 能量过小时保持频率不变

        # 频率排序（保证IMF按频率递增）
        sorted_idx = np.argsort(W)
        W = W[sorted_idx]
        U = U[sorted_idx]

        # 10. 模糊控制调整
        rmse = np.sqrt(np.mean((X - total_recon - e) ** 2))
        rmse_norm = rmse / (np.std(X) + _EPS)
        Uk_power = np.sum(U ** 2, axis=1)
        freq_conc = np.max(Uk_power) / (np.sum(Uk_power) + _EPS) if np.sum(Uk_power) > 0 else 0
        norm_energy = Uk_power / (np.sum(Uk_power) + _EPS) if np.sum(Uk_power) > 0 else 0
        energy_overlap = -np.sum(norm_energy * np.log(norm_energy + _EPS)) / (np.log(K_curr + _EPS) + _EPS)

        if fuzzy_enable and fuzzy_interval > 0 and it % fuzzy_interval == 0 and it > 0:
            a_scale, k_delta = fuzzy_inference(rmse_norm, freq_conc, energy_overlap)

            # 调整alpha
            alpha_curr = 0.95 * alpha_curr + 0.05 * alpha_curr * a_scale
            alpha_curr = np.clip(alpha_curr, N / fs * 100, N / fs * 1000)

            # 调整K
            if fuzzy_allow_K_change:
                K_new = int(np.clip(K_curr + k_delta, 2, 10))
                if K_new != K_curr:
                    if debug:
                        print(f"[FUZZY] it={it}: K {K_curr}→{K_new}, alpha={alpha_curr:.3e}")

                    # 扩展K
                    if K_new > K_curr:
                        extra = []
                        for _ in range(K_new - K_curr):
                            new_freq = W[-1] + fs / (2 * K_new)
                            extra_imf = np.zeros(N)  # 新分量初始化为0，后续迭代会更新
                            extra.append(extra_imf)
                        U = np.vstack([U, extra])
                        W = np.append(W, [W[-1] + fs / (2 * K_new) for _ in range(K_new - K_curr)])

                    # 缩减K（保留能量最大的分量）
                    else:
                        energy = np.sum(U ** 2, axis=1)
                        keep_idx = np.argsort(energy)[-K_new:]  # 保留能量最大的K_new个
                        U = U[keep_idx]
                        W = W[keep_idx]

                    K_curr = K_new
                    # 重新排序
                    sorted_idx = np.argsort(W)
                    W = W[sorted_idx]
                    U = U[sorted_idx]

            last_fuzzy_iter = it

        # 调试输出
        if debug and (it < 10 or it % 20 == 0):
            print(f"[it={it:03d}] rmse={rmse:.4e}, α={alpha_curr:.3f}, K={K_curr}, "
                  f"W=[{W[0]:.0f}, ..., {W[-1]:.0f}]")

        # 收敛检查
        if it - last_fuzzy_iter > 5 and len(rmse_list) > 3:
            rel_change = abs(rmse_list[-1] - rmse) / (rmse + _EPS)
            if rel_change < tol:
                if debug:
                    print(f"✅ 收敛于 iter={it}, Δ={rel_change:.3e}")
                break

        # 记录迭代过程
        rmse_list.append(rmse)
        if record_alpha_K:
            alpha_list.append(alpha_curr)
            K_list.append(K_curr)

    # 结果整理
    result = {
        "alpha_list": alpha_list,
        "K_list": K_list,
        "rmse_list": rmse_list,
        "final_rmse": rmse,
        "iterations": it + 1,
        "energy_ratio": np.sum(np.sum(U ** 2, axis=0)) / (np.sum(X ** 2) + _EPS)  # IMF总能量占比
    }
    return U, W, e, result


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

