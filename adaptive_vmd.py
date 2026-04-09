import os
import numpy as np
from tqdm import tqdm
from config import CFG
from pathlib import Path
from preprocessing import preprocess
from segmentation import frame_signal, ola_reconstruct, ola_selfcheck
from dataio import load_wav, list_wavs_with_labels
from concurrent.futures import ProcessPoolExecutor, as_completed
from rmsa_vmd import estimate_K_by_spectral_peaks, alpha_from_SE, admm_ul_rmsa_vmd, cross_segment_frequency_lock

# 准备函数
def snr_db(x, xhat):
    err = x - xhat
    return 10.0 * np.log10((np.sum(x**2) + 1e-12) / (np.sum(err**2) + 1e-12))

def energy_retention(s, shat):
    # ER = ||ŝ||^2 / ||s||^2
    return float((np.sum(shat**2) + 1e-12) / (np.sum(s**2) + 1e-12))
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

# 分解函数
def RMSA_vmd(seg, fs):
    print(f"\n[DEBUG] 输入段长度: {len(seg)}")
    # 1) K/alpha/lambda 自适应
    # ↑ 通过谱峰估计 IMF 数目，限定最大值为 CFG.vmd.K_max（需保证最小 >= 1）     rmsa_vmd.py
    K = estimate_K_by_spectral_peaks(seg, fs, CFG.vmd.K_max)
    #print(f"[DEBUG] 估计 IMF 数 K={K}")

    # ↑ 依据谱熵（SE）调整 VMD 惩罚因子 alpha（噪声越强 alpha 越大，抑制带宽）    rmsa_vmd.py
    alpha = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)

    # ↑ 试图计算 MAD 估计噪声尺度，但这里**有个小 bug**：
    sigma = np.median(np.abs(seg - np.median(seg))) * 1.4826
    #print(f"[DEBUG] MAD sigma={sigma:.6f}")

    # ↑ 随机采样一个系数 c（区间来自配置），用于控制 L1 稀疏强度
    c = np.random.uniform(*CFG.vmd.lambda_c_range)

    # ↑ L1 正则的阈值（越大则更强稀疏），与噪声尺度成正比
    lam1 = c * sigma

    print(f"估计 IMF 数={K}, alpha={alpha:.4f}, lam1={lam1:.4f}")

    # 2) UL-RMSA-VMD (单段)：返回 U, W, e                               rmsa_vmd.py
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
    # print(f"  VMD分解结果U: {U}")
    # print(f"  VMD分解结果e: {e}")
    # print(f"  VMD分解结果W: {W}")
    print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    return U, W, e, K, alpha, lam1

# 多线程函数
def _process_single_segment(args):
    seg_id, seg, fs = args
    try:
        U, W, e, K, alpha, lam1 = RMSA_vmd(seg, fs)
        #U, W, e, K_used, alpha_used, lam1, meta = RMSA_vmd(seg, fs)
        #print(U)
        # 可能你希望做 U_fix/e_fix：比如去掉全零行或按 K_used 截断
        U_fix, e_fix, v, info = vmd_consistency_and_rescale(U, e, seg)
        print(f"U_fix:{U_fix}")
        print(f"e_fix:{e_fix}")
        print(f"  [一致性] 试探SNR={ {k: f'{v:.2f}' for k, v in info['snr_cands'].items()} } "
              f"=> best={info['best']}, gain={info['gain']:.3f}, "
              f"SNR*={info['snr_after']:.2f} dB, ER*={info['er_after']:.3f}")
        return {
            "ok": True,
            "seg_id": seg_id,
            "U_fix": U_fix,
            "e_fix": e_fix,
            "seg": seg,  # 保存原始段
            "K": int(K),
            "alpha": float(alpha)
        }
    except Exception as ex:
        print(f"[ERROR] Segment {seg_id} 分解失败: {ex}")
        return {"ok": False, "seg_id": seg_id, "error": str(ex)}

# 自适应分解的主函数
def step1_vmd_decompose(data_root: str, vmd_save_dir: str, max_workers: int = 6):
    """
    对输入音频文件执行 RMSA-VMD 分解，并保存 IMF 结果。
    保存内容包括：
    - U: IMF矩阵
    - fs: 采样率
    - alpha, K: 模态参数
    """
    os.makedirs(vmd_save_dir, exist_ok=True)
    # 1. 加载数据列表                               dataio.py
    items = list_wavs_with_labels(data_root)
    assert len(items) > 0, "数据目录为空或未按预期组织"
    print(f"[批量模式] 共加载 {len(items)} 个文件")

    fs = CFG.seg.fs_target  # 目标采样率

    #  2. 逐个文件处理：预处理→分帧→VMD分解→保存
    for i, (path, y) in enumerate(items, 1):
        wav_stem = Path(path).stem
        # 1）VMD结果保存路径（每个文件一个npz）
        vmd_save_path = os.path.join(vmd_save_dir, f"{wav_stem}_label{y}_vmd.npz")

        if os.path.exists(vmd_save_path):
            print(f"⚙️ 跳过已存在VMD结果: {vmd_save_path}")
            continue

        print(f"\n[{i}/{len(items)}] 正在处理: {wav_stem}")

        # 2） 读取 wav 并重采样到 fs        dataio.py
        x_raw = load_wav(path, fs)

        # 3） 去直流、归一化、可选降采样/带通等   preprocessing.py
        x = preprocess(x_raw)

        # 4） 分帧     segmentation.py
        frames, idxs, win, hop = frame_signal(x, fs, CFG.seg.seg_len_s, CFG.seg.hop_ratio, CFG.seg.window)
        n_frames = len(frames)
        print(f"分帧: 共 {len(frames)} 段, frame_len={frames.shape[1]}, hop={hop}")

        tasks = [(seg_id, frames[seg_id], fs) for seg_id in range(n_frames)]
        seg_vmd_results, all_alpha, all_K = [], [], []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_process_single_segment, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=n_frames, desc=f"VMD分解({wav_stem})"):
                res = fut.result()
                if res["ok"]:
                    seg_vmd_results.append(res)
                    all_alpha.append(res["alpha"])
                    print(f"all_alpha:{all_alpha}")
                    all_K.append(res["K"])
                    print(f"all_K:{all_K}")
                else:
                    print(f"⚠️ 段 {res['seg_id']} 分解失败: {res['error']}")

        W_list = [r["freqs"] for r in seg_vmd_results if r.get("freqs") is not None]
        if len(W_list) > 0:
            K_init = int(np.median(all_K))  # 用中位数作为聚类数的参考
            global_centers = cross_segment_frequency_lock(W_list, n_clusters=K_init)
            print(f"[频率锁定] 得到全局频带中心: {np.round(global_centers, 2)} Hz")
        else:
            global_centers = None

        seg_vmd_results.sort(key=lambda x: x["seg_id"])
        all_U = [r["U_fix"] for r in seg_vmd_results if r["ok"]]
        if not all_U:
            print("❌ 无有效IMF结果，跳过保存。")
            continue

        # ===拼接重构===
        K = max(U.shape[0] for U in all_U)
        frame_len = all_U[0].shape[1]
        total_len = frame_len + (len(all_U) - 1) * hop
        U_full = np.zeros((K, total_len))
        e_full = np.zeros(total_len)
        weight_sum = np.zeros(total_len)
        window = win  # Hann窗或其他

        for seg_id, U in enumerate(all_U):
            start = seg_id * hop
            end = start + frame_len
            k_i, _ = U.shape
            # 加窗叠加
            U_full[:k_i, start:end] += U * window[None, :]
            e_full[start:end] += seg_vmd_results[seg_id]["e_fix"] * window
            weight_sum[start:end] += window

        # 归一化（避免重叠区能量放大）
        U_full /= (weight_sum + 1e-12)
        #print(f"U_full:{U_full}")
        e_full /= (weight_sum + 1e-12)
        #print(f"e_full:{e_full}")
        # 每个 IMF 求和后重构整体信号
        full_reconstructed = np.sum(U_full, axis=0)
        #print(f"full_reconstructed:{full_reconstructed}")

        print(f"[IMF组] 形状: {U_full.shape} (模式数×总长度)")
        print(f"[信号重构] 形状: {full_reconstructed.shape}")

        # 对齐长度
        # min_len = min(len(x), len(full_reconstructed))
        # print(f"min_len:{min_len}")
        # x_aligned = x[:min_len]
        # print(f"x_aligned:{x_aligned}")
        # full_reconstructed_aligned = full_reconstructed[:min_len]
        # print(f"full_reconstructed_aligned:{full_reconstructed_aligned}")
        # e_aligned = x_aligned - full_reconstructed_aligned
        # # 计算指标
        # snr_val = snr_db(x_aligned, full_reconstructed_aligned)
        # er_val = energy_retention(x_aligned, full_reconstructed_aligned)
        # residual_energy_ratio = float(np.sum(e_aligned ** 2) / (np.sum(x_aligned ** 2) + 1e-12))
        #
        # print(f"[VMD Quality] SNR={snr_val:.2f} dB | ER={er_val:.4f} | Residual={residual_energy_ratio:.4e}")
        # 保存当前文件的所有VMD结果
        np.savez(
            vmd_save_path,
            wav_path=path,
            label=y,
            fs=fs,
            x_prep=x,      # 预处理后
            frame_win=win,
            frame_hop=hop,
            seg_vmd_results=np.array([
                dict(seg_id=r["seg_id"], U_fix=r["U_fix"], e_fix=r["e_fix"],seg=r["seg"],
                     K=r["K"], alpha=r["alpha"]) for r in seg_vmd_results if r["ok"]
            ], dtype=object),  # ✅ 更安全的保存方式
            alpha_list=np.array(all_alpha),
            K_list=np.array(all_K),
            U_full=U_full,  # ✅ IMF组
            e_full=e_full,  # ✅ 残差拼接信号
            x_recon=full_reconstructed,  # ✅ 重构信号
            global_centers=global_centers  # ✅ 新增 频带对齐？
        )