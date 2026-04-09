# main_pipeline.py
# 主函数
import os
import re
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from dataio import load_wav, list_wavs_with_labels
from preprocessing import preprocess
from segmentation import frame_signal, ola_reconstruct, ola_selfcheck
from rmsa_vmd import estimate_K_by_spectral_peaks, alpha_from_SE, admm_ul_rmsa_vmd, cross_segment_frequency_lock
from IMF_postprocess import prune_merge_imfs, compute_weights, reconstruct
from features import imf_features, mfcc_features
from models_attn_fuser import IMFTokenFuser
from train_contrastive import InfoNCELoss
from train_classifier import crossval_train_eval
from metrics import bo_objective
from config import CFG
from pathlib import Path
from IMF_FBE_select import select_imfs_by_FBE

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
def extract_segment_features(seg, fs, phys_bands):
    """
        对单个语音/振动段 seg（1D numpy 向量）执行：
        1) 参数自适应（K、alpha、lambda）
        2) UL-RMSA-VMD 分解
        3) IMF 剪枝与去噪权重计算 + 重构
        4) 提 IMF-token 特征与 IMF 中心频率
        返回：
          tokens: (L, d_token)  每个 IMF 一条 token
          f_center: (L,)       每个 IMF 的中心频率（能量质心）
          s_hat: (len(seg),)   去噪重构的时域信号
          (U_den, w, e): 去噪后的 IMF 们、权重、残差信号
        """
    # 1) K/alpha/lambda 自适应
    # ↑ 通过谱峰估计 IMF 数目，限定最大值为 CFG.vmd.K_max（需保证最小 >= 1）  rmsa_vmd.py
    K = estimate_K_by_spectral_peaks(seg, fs, CFG.vmd.K_max)

    # ↑ 依据谱熵（SE）调整 VMD 惩罚因子 alpha（噪声越强 alpha 越大，抑制带宽）  rmsa_vmd.py
    alpha = alpha_from_SE(seg, fs, CFG.vmd.alpha0, CFG.vmd.beta_se)

    # ↑ 试图计算 MAD 估计噪声尺度，但这里**有个小 bug**：
    sigma = np.median(np.abs(seg - np.median(seg))) * 1.4826
    # ↑ 随机采样一个系数 c（区间来自配置），用于控制 L1 稀疏强度
    c = np.random.uniform(*CFG.vmd.lambda_c_range)
    # ↑ L1 正则的阈值（越大则更强稀疏），与噪声尺度成正比
    lam1 = c * sigma
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
    print(f"  VMD分解结果: U.shape={U.shape}, W.shape={W.shape}, e.shape={e.shape}")
    U, e, rec_dbg, dbg = vmd_consistency_and_rescale(U, e, seg)
    print(f"  [一致性] 试探SNR={ {k: f'{v:.2f}' for k, v in dbg['snr_cands'].items()} } "
          f"=> best={dbg['best']}, gain={dbg['gain']:.3f}, "
          f"SNR*={dbg['snr_after']:.2f} dB, ER*={dbg['er_after']:.3f}")
    print(f"  [能量分配] ||sumU||^2/||seg||^2={dbg['frac']['sumU/seg']:.3f}, "
          f"||e||^2/||seg||^2={dbg['frac']['e/seg']:.3f}")

    # 3) 模态剪枝与去噪权重   IMF_postprocess.py
    # ↑ 基于能量/相似性做 IMF 剪枝与合并，去除弱/冗余 IMF     剪枝与合并
    if getattr(CFG.post, "use_fbe", False):  # 新增配置开关
        U, fbe_vals = select_imfs_by_FBE(U, fs, use_envelope=True, top_k=CFG.post.fbe_top_k)
        print(f"  [FBE对比筛选] IMF数={U.shape[0]}")
    else:
        U = prune_merge_imfs(U, CFG.post.prune_energy_thr, CFG.post.prune_corr_thr)
        print(f"  [标准剪枝] IMF数={U.shape[0]}")


    # 参考基线（未去噪，直接 sum(U)，可选 +e）
    s_sum = U.sum(axis=0)
    a_sum = np.dot(seg, s_sum) / (np.dot(s_sum, s_sum) + 1e-12)
    s_sum = a_sum * s_sum
    print(f"  [BASELINE sum(U) after LS] SNR={snr_db(seg, s_sum):.2f} dB, RMSE={rmse(seg, s_sum):.4f}")

    s_sum_e = s_sum + e  # 这里 e 很小，按需保留
    print(f"  [BASELINE sum(U)+e after LS] SNR={snr_db(seg, s_sum_e):.2f} dB, RMSE={rmse(seg, s_sum_e):.4f}")

    # —— 权重：归一化 & 打印 ——
    w = compute_weights(U, e, CFG.post.weight_eta, noise_template=None)
    w = np.asarray(w, float)
    w = np.maximum(w, 1e-6)
    w = w / w.sum()
    print(f"  [weights] min={w.min():.3e}, max={w.max():.3e}, sum={w.sum():.6f}, "
          f"top3 idx={np.argsort(w)[-3:][::-1].tolist()}")

    ek = np.sum(U ** 2, axis=1)
    print("  [IMF能量raw]", ["{:.3e}".format(v) for v in ek])

    # ↑ 重构信号：小波/其他方法重构去噪信号，同时返回去噪后的 IMF 们 U_den（形状仍是 (L, N)） IMF_postprocess.py
    s_hat, U_den = reconstruct(U, e, w, CFG.post.wavelet, CFG.post.wavelet_levels)
    print(f"  去噪重构完成: U_den.shape={U_den.shape}, s_hat.shape={s_hat.shape}")

    # 线性合成
    s_tilde = (w[:, None] * U_den).sum(axis=0)  # 是否 + e 取决于你的定义
    # 最小二乘增益对齐
    #a = np.dot(seg, s_tilde) / (np.dot(s_tilde, s_tilde) + 1e-12)
    rms_seg = np.sqrt(np.mean(seg ** 2))
    rms_tilde = np.sqrt(np.mean(s_tilde ** 2))
    a = 1.0 if rms_tilde < 1e-12 else rms_seg / rms_tilde
    s_hat = a * s_tilde  # 用对齐后的 s_hat 参与一切指标
    print(f"  [gain align] a={a:.6f}, ||seg||={np.linalg.norm(seg):.4f}, ||s_hat||={np.linalg.norm(s_hat):.4f}")

    # —— 指标（统一口径）——
    snr_val = snr_db(seg, s_hat)
    rmse_val = rmse(seg, s_hat)
    er_val = float(np.sum(s_hat ** 2) / (np.sum(seg ** 2) + 1e-12))
    corr = float(np.corrcoef(seg, s_hat)[0, 1])
    print(f"  重构指标(对齐后): SNR={snr_val:.2f} dB, RMSE={rmse_val:.4f}, ER={er_val:.3f}, corr={corr:.4f}")

    # 4) IMF-token 特征
    # ↑ 把每条 IMF 映射成一个 token（如时频能量、带宽、峭度等）    features.py
    # 4) IMF-token 特征融合
    tokens_1 = imf_features(U_den, fs, CFG.phys.bands)

    # 针对每个 IMF 提取 MFCC，并堆叠
    tokens_2 = []
    for k in range(U_den.shape[0]):
        mfcc_vec = mfcc_features(U_den[k], fs, n=20, use_deltas=True)
        tokens_2.append(mfcc_vec)
    tokens_2 = np.vstack(tokens_2)

    # 横向拼接
    tokens = np.hstack([tokens_1, tokens_2])
    # IMF 中心频率估计（由能量质心）
    f_center = []
    for k in range(U_den.shape[0]):
        Uk = np.fft.rfft(U_den[k])
        freqs = np.fft.rfftfreq(len(U_den[k]), 1 / fs)
        P = np.abs(Uk) ** 2 + 1e-12
        f_center.append((freqs * P).sum() / P.sum())
    f_center = np.array(f_center, dtype=float)
    print(f"  特征提取: tokens.shape={tokens.shape}, "f"f_center范围=({f_center.min():.2f}, {f_center.max():.2f})")

    return tokens, f_center, s_hat, (U_den, w, e), snr_val, rmse_val


def run_pipeline(data_root: str,  single_file: str = None):
    os.makedirs("outputs/segments", exist_ok=True)
    os.makedirs("outputs/files_4", exist_ok=True)
    # ↑ 遍历数据根目录，返回 [(path, label), ...]
    # === 1) 数据准备 ===
    if single_file is not None:
        # 只处理单个文件
        items = [(single_file, 0)]  # 注意：第二个元素是标签，这里先写死为 0
        print(f"[单文件模式] {single_file}")
    else:
        # 遍历目录
        items = list_wavs_with_labels(data_root)
        print("样本标签统计：", [y for _, y in items][:20])
        print("N files:", len(items))
        print(items[:3])
        assert len(items) > 0, "数据目录为空或未按预期组织"

    # ↑ 目标采样率
    fs = CFG.seg.fs_target

    all_Z = []  # 融合后的向量 z
    all_Y = []  # 标签
    attn_maps = []   # 保存注意力
    all_W_centers = []   # 每段的 IMF 中心频率集合，便于诊断/画图

    # 轻量注意力融合器
    #d_token = 5 + 2 * len(CFG.phys.bands)  # features.py 中单个IMF-token长度
    #proj = torch.nn.Linear(d_token, CFG.fuser.d_model)
    fuser = IMFTokenFuser(d_model=CFG.fuser.d_model, nhead=CFG.fuser.nhead,
                          num_layers=CFG.fuser.num_layers, dropout=CFG.fuser.dropout,
                          se_ratio=CFG.fuser.se_ratio, bands=CFG.phys.bands)
    fuser.eval()
    proj = None  # 延迟初始化

    for i, (path, y) in enumerate(items, 1):
        out_path = f"outputs/files_4/{Path(path).stem}_label{y}.npz"
        if os.path.exists(out_path):
            print(f"⚙️ 跳过已存在文件: {out_path}")
            continue
        # ↑ 读取 wav 并重采样到 fs；返回 1D numpy 数组
        print(f"\n--- 处理文件 {i}/{len(items)}: {path}, 标签={y} ---")
        x = load_wav(path, fs)         # dataio.py
        print(f"加载完成: x.shape={x.shape}, fs={fs}")
        # ↑ 去直流、归一化、可选降采样/带通等
        x = preprocess(x)              # preprocesssing.py
        print(f"预处理完成: mean={x.mean():.4f}, std={x.std():.4f}")
        # ↑ 把整段 x 切成重叠的短段（窗口/步长由配置决定）     segmentation.py
        frames, idxs, win, hop = frame_signal(x, fs, CFG.seg.seg_len_s, CFG.seg.hop_ratio, CFG.seg.window)
        print(f"分帧: 共 {len(frames)} 段, frame_len={frames.shape[1]}, hop={hop}")

        # 对每个段提取 UL-RMSA-VMD + token
        Z_segments = []
        file_attn = []
        s_hat_list = []
        U_dens_all = []

        for seg in frames:
            tokens, f_center, s_hat, (U_den, w, e), snr_val, rmse_val = extract_segment_features(seg, fs, CFG.phys.bands)
            s_hat_list.append(s_hat)
            # 确保 U_den 是二维数组
            if U_den.ndim == 2:
                U_dens_all.append(U_den)
            else:
                print(f"⚠️ U_den 维度异常: {U_den.shape}")
            # ===== 融合处理 =====
            # 第一次自动推断 token 维度
            if proj is None:
                d_token = tokens.shape[1]
                proj = torch.nn.Linear(d_token, CFG.fuser.d_model)
                print(f"  [自动推断] token维度={d_token}, 创建 Linear({d_token}, {CFG.fuser.d_model})")
            T = torch.tensor(tokens, dtype=torch.float32).unsqueeze(0)
            T = proj(T)
            F = torch.tensor(f_center, dtype=torch.float32).view(1, -1, 1)
            with torch.no_grad():
                z, h = fuser(T, F)
            Z_segments.append(z.numpy().squeeze(0))
            file_attn.append(h.mean(dim=-1).squeeze(0).numpy())
            all_W_centers.append(f_center)

        s_hat_frames = np.stack(s_hat_list, axis=0)  # shape = (num_frames, frame_len)
        s_hat_full = ola_reconstruct(s_hat_frames, win, hop, frames_are_windowed=True)
        print(f"[OLA check] frames={s_hat_frames.shape}, win={win.shape}, hop={hop}, "
              f"min(wsum-like)≈{np.min(np.convolve(win, np.ones_like(win))[::hop]) if hop > 0 else 'N/A'}")
        # IMF拼接处理

        # === 拼接所有帧的 IMF ===
        if len(U_dens_all) > 0:
            max_k = max(U.shape[0] for U in U_dens_all)
            U_dens_padded = []
            for U in U_dens_all:
                K, N = U.shape
                if K < max_k:
                    pad = np.zeros((max_k - K, N))
                    U = np.vstack([U, pad])
                U_dens_padded.append(U)
            U_dens_all = np.concatenate(U_dens_padded, axis=1)
            print(f"[IMF 拼接] 已填充至统一形状: {U_dens_all.shape}")
        else:
            U_dens_all = None
            print("⚠️ 无有效 IMF 数据。")
        # 文件级池化
        Z_file = np.mean(np.vstack(Z_segments), axis=0)
        all_Z.append(Z_file)
        all_Y.append(y)
        attn_maps.extend(file_attn)

        # ===== 保存文件级结果 =====
        out_path = f"outputs/files_4/{Path(path).stem}_label{y}.npz"
        np.savez(out_path,
                 Z_file=Z_file, label=y,
                 attn_maps=np.array(file_attn, dtype=object),
                 s_hat_full=s_hat_full,
                 U_den=U_dens_all,
                 w = w,
                 e = e,
                 x_raw=x)
        print(f"✅ 保存 {out_path}")

        # === 按原类名保存 ===
        rel_path = os.path.relpath(path, data_root)
        class_dir = os.path.dirname(rel_path)
        # 根据模式选择输出路径
        if single_file is not None:
            out_subdir = os.path.join(os.path.dirname(single_file), class_dir)
        else:
            out_subdir = os.path.join("outputs/files_by_class_4", class_dir)
        os.makedirs(out_subdir, exist_ok=True)
        fname = os.path.splitext(os.path.basename(path))[0] + ".npz"
        np.savez(os.path.join(out_subdir, fname),
                 Z_file=Z_file,
                 label=int(y),
                 fname=path)

    print(f"[特征提取完成] 共保存 {len(items)} 个样本 -> {single_file}")

import collections
def train_model(model_name="svm", cv_folds=None, feature_dir="outputs/files_by_class"):
    #files = sorted([f for f in os.listdir(feature_dir) if f.endswith(".npz")])
    files = sorted(Path(feature_dir).rglob("*.npz"))
    assert files, f"未发现特征文件，请确认 {feature_dir} 是否存在。"

    all_Z, all_Y = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        all_Z.append(d["Z_file"])
        all_Y.append(int(d["label"]))
    X = np.vstack(all_Z)
    y = np.array(all_Y)
    print(f"[加载训练数据] X.shape={X.shape}, y.shape={y.shape}")
    print("[类别分布]", collections.Counter(y))

    if cv_folds is None:
        cv_folds = CFG.bayes.cv_folds

    f1, aurc, y_true, y_pred, clf = crossval_train_eval(
        X, y, model_name=model_name, n_splits=cv_folds, return_preds=True
    )

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


if __name__ == "__main__":
    # 示例：python main_pipeline.py 之前，请确保数据目录结构正确
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"  # 替换为你的数据根目录
    run_pipeline(data_root)
    class_map = {
        "A": 0, "B": 1, "C": 2, "D": 3, "E": 4,
        "F": 5, "G": 6, "H": 7, "I": 8, "J": 9
    }
    #reclassify_saved_files("outputs/files", "outputs/files_reclassified", class_map=class_map)
    #train_model(model_name="svm", feature_dir="outputs/files_by_class_4")
    # 手动替换分类器
    train_model(model_name="rf", feature_dir="outputs/files_by_class_4")
    #train_model(model_name="mlp", feature_dir="outputs/files_reclassified")

    # 先跑单文件
    #single_file = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data/A/filenamez1.wav"
    #run_pipeline(data_root, single_file=single_file)

    ola_selfcheck()

