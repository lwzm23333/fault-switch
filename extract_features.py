import os
import re
import pandas as pd
import numpy as np
from config import CFG
from pathlib import Path
from IMF_FBE_select import select_imfs_by_FBE
from models_attn_fuser import IMFTokenFuser
from IMF_postprocess import prune_merge_imfs, compute_weights, reconstruct
from features_all import imf_features, FEATURE_LIST
from joblib import Parallel, delayed

def step2_extract_features_plain(
        vmd_save_dir: str,
        feat_excel_path: str,
        feat_npz_path: str
):
    """
    Step2: 进行工程特征构建
      - IMF 模态筛选
      - 加权重构
      - 多域 IMF token 30维特征提取
      - 文件级聚合
    输出：
      Excel：文件级特征 （方便查看）
      npz：用于 step3 的原始特征矩阵
    """
    os.makedirs(os.path.dirname(feat_excel_path), exist_ok=True)
    os.makedirs(feat_npz_path, exist_ok=True)

    # 1. 加载所有VMD结果文件
    vmd_files = [f for f in os.listdir(vmd_save_dir) if f.endswith("_vmd.npz")]
    assert len(vmd_files) > 0, f"未找到VMD结果文件，请检查 {vmd_save_dir}"
    print(f"共加载 {len(vmd_files)} 个VMD结果文件")

    all_file_feats = []
    all_labels = []
    all_file_names = []

    # 2. 逐个VMD结果文件处理
    for f in vmd_files:
        vmd_file_path = os.path.join(vmd_save_dir, f)
        # 1）解析文件名中的标签和原始文件名
        label_match = re.search(r"label(\d+)_vmd\.npz", f)
        assert label_match, f"VMD文件名格式错误: {f}"
        label = int(label_match.group(1))

        # 3）加载VMD结果
        vmd_data = np.load(vmd_file_path, allow_pickle=True)
        seg_results = vmd_data["seg_vmd_results"].tolist()
        fs = int(vmd_data.get("fs", CFG.seg.fs_target))
        stem = f.replace(f"_label{label}_vmd.npz", "")

        print(f"\n--- 文件: {stem}, 标签={label}, 段数={len(seg_results)} ---")

        seg_feat_list, s_hat_list, U_dens_all = [], [], []

        # 4）逐段特征提取
        for seg_idx, seg_res in enumerate(seg_results):
            U = seg_res["U_fix"]
            e = seg_res["e_fix"]

            # ---------- 模态筛选 ----------
            if getattr(CFG.post, "use_fbe", False):      # IMF_FBE_select.py
                U_sel, _ = select_imfs_by_FBE(U, fs, use_envelope=True, top_k=CFG.post.fbe_top_k)
            else:                                        # IMF_postprocess.py
                U_sel = prune_merge_imfs(U, CFG.post.prune_energy_thr, CFG.post.prune_corr_thr)
            if U_sel.size == 0:
                print(f"⚠️ 段{seg_idx} IMF为空，跳过。")
                continue

            # ---------- 加权重构 ----------
            w = compute_weights(U_sel, e, CFG.post.weight_eta)         # IMF_postprocess.py
            w = np.maximum(w, 1e-6)
            w /= w.sum()
            s_hat, U_den = reconstruct(U_sel, e, w, CFG.post.wavelet, CFG.post.wavelet_levels)   # IMF_postprocess.py
            s_hat_list.append(s_hat)
            U_dens_all.append(U_den)

            # # RMS增益对齐  这部分貌似不需要
            # # 线性合成
            # # s_tilde = (w[:, None] * U_den).sum(axis=0)  # 是否 + e 取决于你的定义
            # rms_seg = np.sqrt(np.mean(seg ** 2))
            # rms_tilde = np.sqrt(np.mean(s_hat ** 2))
            # a = 1.0 if rms_tilde < 1e-12 else rms_seg / rms_tilde
            # s_hat = a * s_hat

            # # 4）指标
            # snr_val = snr_db(seg, s_hat)
            # rmse_val = rmse(seg, s_hat)
            # er_val = float(np.sum(s_hat ** 2) / (np.sum(seg ** 2) + 1e-12))
            # corr = float(np.corrcoef(seg, s_hat)[0, 1])
            # print(f"  重构指标(对齐后): SNR={snr_val:.2f} dB, RMSE={rmse_val:.4f}, ER={er_val:.3f}, corr={corr:.4f}")

            # ---------- IMF feature（30+维） ----------  features_all.py
            feats, feat_names = imf_features(U_den, fs, CFG.phys.bands)

            # ---------- 将每段 IMF 特征池化成一个向量 ----------
            seg_feat = np.mean(feats, axis=0)
            seg_feat_list.append(seg_feat)

        if len(seg_feat_list) == 0:
            print(f"⚠ 文件 {stem} 无有效 IMF 特征")
            continue

        # 5）文件级聚合
        file_feat = np.mean(np.vstack(seg_feat_list), axis=0)

        # 保存单文件 npz
        class_dir = os.path.join(feat_npz_path, f"class_{label}")
        os.makedirs(class_dir, exist_ok=True)
        file_feat_path = os.path.join(class_dir, f"{stem}_label{label}_feat.npz")

        np.savez(file_feat_path, X=file_feat, y=label, filename=stem)
        print(f"📦 单文件特征已保存: {file_feat_path}")
        all_file_feats.append(np.load(file_feat_path)["X"])
        all_labels.append(label)
        all_file_names.append(stem)


    # 6）保存 Excel
    all_file_feats_list = []  # 每行存 dict 而不是 ndarray
    for f_idx, file_feat in enumerate(all_file_feats):
        feat_dict = {"filename": all_file_names[f_idx], "label": all_labels[f_idx]}
        for feat_name, val in zip(feat_names, file_feat):
            feat_dict[feat_name] = val  # 多维特征保持 list
        all_file_feats_list.append(feat_dict)

    df = pd.DataFrame(all_file_feats_list)
    df.to_csv(feat_excel_path.replace(".xlsx", ".csv"), index=False)

    # 7）保存统一 npz
    unified_npz_path = os.path.join(feat_npz_path, "all_file_features.npz")
    np.savez(unified_npz_path,
             X=np.vstack(all_file_feats),
             y=np.array(all_labels),
             filename=np.array(all_file_names))
    print(f"📦 统一特征矩阵已保存至 {unified_npz_path}")

    return df
