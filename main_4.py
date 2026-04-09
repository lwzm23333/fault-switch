import os
import re
import torch
import librosa
import collections
import numpy as np
import pandas as pd
from typing import Tuple, Dict
from tqdm import tqdm
from config import CFG
from pathlib import Path
from metrics import bo_objective
from preprocessing import preprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from segmentation import frame_signal, ola_reconstruct, ola_selfcheck
#from fuzzy_rmsa_vmd import estimate_K_by_spectral_peaks, alpha_from_SE, admm_ul_rmsa_vmd, cross_segment_frequency_lock
from rmsa_vmd import estimate_K_by_spectral_peaks, alpha_from_SE, admm_ul_rmsa_vmd, cross_segment_frequency_lock
from dataio import load_wav, list_wavs_with_labels, add_awgn
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from train_classifier import crossval_train_eval
from sklearn.model_selection import train_test_split
from IMF_FBE_select import select_imfs_by_FBE
from models_attn_fuser import IMFTokenFuser
from IMF_postprocess import prune_merge_imfs, compute_weights, reconstruct
from features import imf_features, mfcc_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC
from features_all import feature_selection, FEATURE_LIST
from scipy.signal import welch
from scipy.stats import entropy
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
    print(f"\n[DEBUG] 输入段长度: {len(seg)}")
    # 1) K/alpha/lambda 自适应
    # ↑ 通过谱峰估计 IMF 数目，限定最大值为 CFG.vmd.K_max（需保证最小 >= 1）  rmsa_vmd.py
    K = estimate_K_by_spectral_peaks(seg, fs, CFG.vmd.K_max)
    print(f"[DEBUG] 估计 IMF 数 K={K}")

    # ↑ 依据谱熵（SE）调整 VMD 惩罚因子 alpha（噪声越强 alpha 越大，抑制带宽）  rmsa_vmd.py
   # alpha, SE = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)
    alpha = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)

    # ↑ 试图计算 MAD 估计噪声尺度，但这里**有个小 bug**：
    sigma = np.median(np.abs(seg - np.median(seg))) * 1.4826
    print(f"[DEBUG] MAD sigma={sigma:.6f}")
    # ↑ 随机采样一个系数 c（区间来自配置），用于控制 L1 稀疏强度
    c = np.random.uniform(*CFG.vmd.lambda_c_range)
    # ↑ L1 正则的阈值（越大则更强稀疏），与噪声尺度成正比
    lam1 = c * sigma
    print(f"[DEBUG] lam1={lam1:.6f} (随机系数 c={c:.4f})")
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
    # print(f"  VMD分解结果U: {U}")
    # print(f"  VMD分解结果e: {e}")
    # print(f"  VMD分解结果W: {W}")
    print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    return U, W, e, K, alpha, lam1
    #  使用fuzzy_rmsa_vmd.py的时候
    # U, W, e, meta = admm_ul_rmsa_vmd(seg, fs, K, alpha,
    #                                  lam1=lam1,
    #                                  lam2=CFG.vmd.gamma_bw,
    #                                  lam3=CFG.vmd.gamma_w_smooth,
    #                                  lam4=CFG.vmd.gamma_lock,
    #                                  iters=CFG.vmd.admm_iters,
    #                                  tol=CFG.vmd.admm_tol,
    #                                  seed=CFG.vmd.seed)
    # K_used = meta.get("K", K)
    # alpha_used = meta.get("alpha", alpha)
    # print(f"  VMD分解结果U: {U}")
    # print(f"  VMD分解结果e: {e}")
    # print(f"  VMD分解结果W: {W}")
    # print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    # return U, W, e, K_used, alpha_used, lam1, meta

def _process_single_segment(args):
    seg_id, seg, fs = args
    try:
        U, W, e, K, alpha, lam1 = RMSA_vmd(seg, fs)
        #U, W, e, K_used, alpha_used, lam1, meta = RMSA_vmd(seg, fs)
        print(U)
        # 可能你希望做 U_fix/e_fix：比如去掉全零行或按 K_used 截断
        U_fix, e_fix, v, info = vmd_consistency_and_rescale(U, e, seg)
        print(f"U_fix:{U_fix}")
        print(f"e_fix:{e_fix}")
        print(f"  [一致性] 试探SNR={ {k: f'{v:.2f}' for k, v in info['snr_cands'].items()} } "
              f"=> best={info['best']}, gain={info['gain']:.3f}, "
              f"SNR*={info['snr_after']:.2f} dB, ER*={info['er_after']:.3f}")
        print(f"  [能量分配] ||sumU||^2/||seg||^2={info['frac']['sumU/seg']:.3f}, "
              f"||e||^2/||seg||^2={info['frac']['e/seg']:.3f}")
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
def step1_vmd_decompose(data_root: str, vmd_save_dir: str, max_workers: int = 6):
    """
    对输入音频文件执行 RMSA-VMD 分解，并保存 IMF 结果。
    保存内容包括：
    - U: IMF矩阵
    - fs: 采样率
    - alpha, K: 模态参数
    """
    os.makedirs(vmd_save_dir, exist_ok=True)
    # 1. 加载数据列表
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

        print(f"\n[{i}/{len(items)}] 正在处理: {wav_stem}")

        # ↑ 读取 wav 并重采样到 fs
        x_raw = load_wav(path, fs)
        # ↑ 去直流、归一化、可选降采样/带通等
        x = preprocess(x_raw)

        # ↑ 分帧
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
        all_e = [r["e_fix"] for r in seg_vmd_results if r["ok"]]
        all_seg = [r["seg"] for r in seg_vmd_results if r["ok"]]
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
        print(f"U_full:{U_full}")
        e_full /= (weight_sum + 1e-12)
        print(f"e_full:{e_full}")
        # 每个 IMF 求和后重构整体信号
        full_reconstructed = np.sum(U_full, axis=0)
        print(f"full_reconstructed:{full_reconstructed}")

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
            x_raw=x_raw,   # 原始音频/经由重采样，但是这里好像没有
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

# ---------------------------------------------------
# Step 2: 从分解结果中提取重构信号特征
# ---------------------------------------------------
def step2_extract_features(vmd_save_dir: str, feat_excel_path: str):
    """
    从VMD结果目录读取分解结果（U、e、α、K等），执行模态筛选、加权重构、特征提取与融合。
    输出整合的 Excel 特征文件，方便后续 Step3 特征筛选。
    """
    os.makedirs(os.path.dirname(feat_excel_path), exist_ok=True)

    vmd_files = [f for f in os.listdir(vmd_save_dir) if f.endswith("_vmd.npz")]
    assert len(vmd_files) > 0, f"未找到VMD结果文件，请检查 {vmd_save_dir}"
    print(f"共加载 {len(vmd_files)} 个 VMD 结果文件")

    # 初始化融合器
    fuser = IMFTokenFuser(
        d_model=CFG.fuser.d_model, nhead=CFG.fuser.nhead,
        num_layers=CFG.fuser.num_layers, dropout=CFG.fuser.dropout,
        se_ratio=CFG.fuser.se_ratio, bands=CFG.phys.bands
    )
    fuser.eval()
    proj = None  # 延迟初始化映射层

    all_records = []  # 存储每个文件级特征记录

    # ====== 遍历文件 ======
    for f in vmd_files:
        vmd_file_path = os.path.join(vmd_save_dir, f)
        label_match = re.search(r"label(\d+)_vmd\.npz", f)
        assert label_match, f"文件名格式错误: {f}"
        label = int(label_match.group(1))
        wav_stem = f.replace(f"_label{label}_vmd.npz", "")

        print(f"\n--- 提取特征: {wav_stem}, 标签={label} ---")

        vmd_data = np.load(vmd_file_path, allow_pickle=True)
        fs = int(vmd_data.get("fs", CFG.seg.fs_target))
        frame_win = vmd_data["frame_win"]
        frame_hop = int(vmd_data["frame_hop"])
        seg_vmd_results = vmd_data["seg_vmd_results"].tolist()
        alpha_list = np.array(vmd_data["alpha_list"])
        K_list = np.array(vmd_data["K_list"])

        print(f"  段数={len(seg_vmd_results)}, α范围=({alpha_list.min():.2f},{alpha_list.max():.2f}), K均值={K_list.mean():.2f}")

        seg_feats, seg_attn, s_hat_list, U_dens_all = [], [], [], []

        # ====== 逐段处理 ======
        for seg_idx, seg_res in enumerate(seg_vmd_results):
            U = seg_res["U_fix"]
            e = seg_res["e_fix"]

            # 模态筛选
            if getattr(CFG.post, "use_fbe", False):
                U_sel, _ = select_imfs_by_FBE(U, fs, use_envelope=True, top_k=CFG.post.fbe_top_k)
            else:
                U_sel = prune_merge_imfs(U, CFG.post.prune_energy_thr, CFG.post.prune_corr_thr)
            if U_sel.size == 0:
                print(f"⚠️ 段{seg_idx} IMF为空，跳过。")
                continue

            # 加权重构
            w = compute_weights(U_sel, e, CFG.post.weight_eta)
            w = np.maximum(np.asarray(w, float), 1e-6)
            w /= w.sum()
            s_hat, U_den = reconstruct(U_sel, e, w, CFG.post.wavelet, CFG.post.wavelet_levels)
            s_hat_list.append(s_hat)
            U_dens_all.append(U_den)

            # 提取 IMF token 特征
            tokens_1 = imf_features(U_den, fs, CFG.phys.bands)
            tokens_2 = np.vstack([
                librosa.feature.mfcc(y=U_den[k], sr=fs, n_mfcc=20).mean(axis=1)
                for k in range(U_den.shape[0])
            ])
            tokens = np.hstack([tokens_1, tokens_2])

            # 中心频率
            f_center = []
            for k in range(U_den.shape[0]):
                Uk = np.fft.rfft(U_den[k])
                freqs = np.fft.rfftfreq(len(U_den[k]), 1 / fs)
                P = np.abs(Uk) ** 2 + 1e-12
                f_center.append((freqs * P).sum() / P.sum())
            f_center = np.array(f_center, dtype=float)

            # Transformer 融合
            if proj is None:
                d_token = tokens.shape[1]
                proj = torch.nn.Linear(d_token, CFG.fuser.d_model)
                print(f"  初始化特征映射层: Linear({d_token}, {CFG.fuser.d_model})")

            T = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
            T = proj(T)
            F = torch.tensor(f_center, dtype=torch.float32).view(1, -1, 1)
            with torch.no_grad():
                z, h = fuser(T, F)

            seg_feats.append(z.numpy().squeeze(0))
            seg_attn.append(h.mean(dim=-1).squeeze(0).numpy())

        # ====== 文件级融合 ======
        if not seg_feats:
            print(f"❌ 文件 {wav_stem} 无有效特征，跳过。")
            continue

        file_feat = np.mean(np.vstack(seg_feats), axis=0)

        # 保存记录到 DataFrame
        record = {"file": wav_stem, "label": label}
        for i, fname_feat in enumerate(FEATURE_LIST[:len(file_feat)]):
            record[fname_feat] = file_feat[i]
        all_records.append(record)

    # ====== 汇总并保存为 Excel ======
    df = pd.DataFrame(all_records)
    df.to_excel(feat_excel_path, index=False)
    print(f"\n✅ Step2 特征提取完成，共 {len(df)} 个样本，已保存到：{feat_excel_path}")
    return df


# ---------------------------------------------------
# Step 3: 特征筛选（方差 + 相关 + 互信息）
# ---------------------------------------------------
def step3_feature_selection(excel_path="outputs/features_all.xlsx",
                            label_col="label",
                            top_k=30,
                            save_path="outputs/features_selected.xlsx"):
    """
    对 Step2 提取的特征进行筛选（方差+相关+互信息），并保存为 Excel
    """
    print("\n🚀 Step 3: 特征筛选开始 ...")
    df = pd.read_excel(excel_path)

    df_sel, top_feats = feature_selection(df, label_col=label_col, top_k=top_k)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_sel.to_excel(save_path, index=False)

    print(f"✅ 筛选完成，共保留 {len(top_feats)} 个特征，已保存至 {save_path}")
    print(f"⭐ Top 特征示例: {top_feats[:10]}")
    return df_sel, top_feats


# -----------------------------------------------------------
# Step 4: 融合保存 npz 文件
# -----------------------------------------------------------
def fuse_features_to_npz(selected_features_excel="outputs/features_selected.xlsx",
                         npz_save_dir="outputs/features_npz"):
    """
    将筛选后的特征保存为 npz 文件，每个样本一个 npz，
    方便后续分类训练或 t-SNE/PCA 可视化分析。
    """
    os.makedirs(npz_save_dir, exist_ok=True)

    df = pd.read_excel(selected_features_excel)
    feature_cols = [c for c in df.columns if c not in ["file", "label"]]

    npz_paths = []
    for idx, row in df.iterrows():
        fname = Path(row["file"]).stem
        label = int(row["label"])
        features = row[feature_cols].values.astype(np.float32)

        save_path = os.path.join(npz_save_dir, f"{fname}_label{label}.npz")
        np.savez(save_path,
                 file=fname,
                 label=label,
                 features=features)
        npz_paths.append(save_path)

    print(f"✅ 融合完成，共保存 {len(npz_paths)} 个 npz 文件至 {npz_save_dir}")
    return npz_paths

# -----------------------------------------------------------
# 分类训练函数（修改版）
# -----------------------------------------------------------
def train_model(model_name="svm",
                df=None,
                label_col="label",
                cv_folds=None):
    """
    训练分类器并输出结果
    model_name: "svm" 或 "rf"
    df: pandas DataFrame，包含特征和标签
    """
    assert df is not None, "请传入特征 DataFrame"

    X = df.drop(columns=["file", label_col], errors="ignore").values
    y = df[label_col].values

    print(f"[加载训练数据] X.shape={X.shape}, y.shape={y.shape}")
    print("[类别分布]", collections.Counter(y))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    if model_name.lower() == "svm":
        clf = SVC(kernel="rbf", probability=True, random_state=42)
    elif model_name.lower() == "rf":
        clf = RandomForestClassifier(n_estimators=200, random_state=42)
    else:
        raise ValueError(f"不支持的 model_name: {model_name}")

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None


    if cv_folds is None:
        cv_folds = CFG.bayes.cv_folds

    f1, aurc, y_true, y_pred, clf = crossval_train_eval(
        X, y, model_name=model_name, n_splits=cv_folds, return_preds=True
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

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

def step5_train_classifier(feature_dir="outputs/features", model_name="rf"):
    return train_model(model_name=model_name, feature_dir=feature_dir)

if __name__ == "__main__":
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"
    #data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data_1"
    # === 阶段1：分解 ===
    step1_vmd_decompose(data_root, vmd_save_dir="outputs/imfs_5")

    #df = step2_extract_features(vmd_save_dir="outputs/imfs_5", feat_excel_path="outputs/features_all.xlsx")

    #step3_feature_selection(df, label_col="label", save_path="outputs/features_selected.xlsx")



