import numpy as np
import scipy.signal as sps
from numpy.fft import rfft, irfft, rfftfreq
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

# 在vmd分解中加fuzzy

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
    def _tri_mf(x, a, b, c):
        x = float(x)
        if x <= a or x >= c:
            return 0.0
        if x == b:
            return 1.0
        if x < b:
            return (x - a) / (b - a + 1e-12)
        return (c - x) / (c - b + 1e-12)

    def rms(x):
        return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2))

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
            scale = np.std(np.abs(analytic)) + _EPS
            analytic_norm = analytic / (scale)
            exponent = np.exp(-2j * np.pi * W[k] * t)
            demod = analytic_norm * exponent

            bw = np.clip(W[k] * 0.5 / max(1, K_curr), 20, fs / 10)
            cutoff = float(np.clip(bw / (fs / 2), 0.001, 0.99))
            if cutoff <= 0:
                base_complex = demod
            else:
                b, a = sps.butter(3, cutoff, btype='low')
                real_f = sps.filtfilt(b, a, np.real(demod))
                imag_f = sps.filtfilt(b, a, np.imag(demod))
                base_complex = real_f + 1j * imag_f

            # 4. 最小二乘增益计算（让IMF最佳逼近当前剩余残差）
            denom = np.vdot(base_complex, base_complex).real + _EPS
            g_raw = np.vdot(rem, base_complex) / (denom + _EPS)
            g = float(np.real_if_close(g_raw))
            # 先做温和缩放以避免过大，且把 alpha 作为正则因子（如果希望）
            g = g / (1.0 + (alpha_curr / (np.linalg.norm(base_complex) + _EPS)))
            g = float(np.clip(g, -2.0, 2.0))

            rho = 0.2
            u_new = (g * base_complex).real
            # 防止 NaN/inf
            if not np.all(np.isfinite(u_new)):
                u_new = np.zeros_like(u_new)
            U[k, :] = (1 - rho) * U[k, :] + rho * u_new
            rem = rem - U[k, :]

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
                cent = np.sum(freqs * Uk_power) / (ps_sum + _EPS)

                # 限制频率跳变幅度（避免突变）
                max_shift = 0.1 * fs / K_curr  # 最大移动量：带宽的0.1倍
                W[k] = float(np.clip(cent, W[k] - max_shift, W[k] + max_shift))
            else:
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
            alpha_curr = np.clip(alpha_curr, 1e3, 1e6)

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


def plot_decomposition(x, fs, U, W, e, result, title="自适应VMD分解结果"):
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

    # 收敛曲线和能量占比
    ax_conv = fig.add_subplot(gs[K + 2])
    ax_conv.plot(result["rmse_list"], label="RMSE")
    ax_conv.set_title(f"shoulianquxian (IMF-ENergy: {result['energy_ratio']:.2%})")
    ax_conv.set_xlabel("didaicishu")
    ax_conv.set_ylabel("RMSE")
    ax_conv.grid(True)
    ax_conv.legend()

    plt.tight_layout()
    return fig


# ---------- 使用示例 ----------
if __name__ == "__main__":
    # 生成测试信号（多频率混合）
    fs = 44100
    t = np.arange(0, 0.5, 1 / fs)
    x = (0.5 * np.sin(2 * np.pi * 500 * t) +  # 500Hz
         0.3 * np.sin(2 * np.pi * 1500 * t) +  # 1500Hz
         0.2 * np.sin(2 * np.pi * 3000 * t) +  # 3000Hz
         0.1 * np.random.randn(len(t)))  # 噪声

    # 运行分解（关键参数调整）
    U, W, e, result = admm_ul_rmsa_vmd(
        x,
        fs=fs,
        K=3,  # 初始分量数（与信号频率数匹配）
        alpha=len(x) / fs * 500,  # alpha与信号长度适配
        lam1=0.01,  # 减小稀疏正则化强度（关键）
        iters=200,
        tol=1e-5,
        fuzzy_enable=True,
        fuzzy_interval=15,
        debug=True
    )

    # 可视化结果（重点检查IMF能量和频率差异）
    fig = plot_decomposition(x, fs, U, W, e, result)
    plt.show()