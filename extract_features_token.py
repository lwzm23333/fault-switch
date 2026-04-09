import os
import re
import pandas as pd
import numpy as np
from config import CFG
from pathlib import Path
from IMF_FBE_select import select_imfs_by_FBE
from models_attn_fuser import IMFTokenFuser
from IMF_postprocess import prune_merge_imfs, compute_weights, reconstruct
from features_list import IMF_features_list, FEATURE_LIST
from joblib import Parallel, delayed
import scipy

def imf_to_token(U_single: np.ndarray, fs: float, token_dim: int = 30):
    """
    输入：单个 IMF (1D)
    输出：token_dim 维特征向量
    使用 imf_features() 提取 30 维特征。
    """

    U = U_single.reshape(1, -1)   # 变为 (K=1, N)

    tokens, feat_names = IMF_features_list(
        U=U,
        fs=fs,
        phys_bands=CFG.phys.bands,
        selected_features=FEATURE_LIST   # 已保证长度=30
    )

    token = tokens[0]   # shape (30,)

    # ---- 和 token_dim 对齐（如果未来 FEATURE_LIST != token_dim）----
    if len(token) < token_dim:
        token = np.concatenate([token, np.zeros(token_dim - len(token))])
    elif len(token) > token_dim:
        token = token[:token_dim]

    return token.astype(float)

def process_single_vmd_file(
        f: str,
        vmd_save_dir: str,
        IMF_npz_path: str,
):
    f_path = os.path.join(vmd_save_dir, f)

    label_match = re.search(r"label(\d+)_vmd\.npz", f)
    if not label_match:
        print(f"❌ 文件名标签解析失败: {f}")
        return None

    label = int(label_match.group(1))
    stem = f.replace(f"_label{label}_vmd.npz", "")

    class_dir = os.path.join(IMF_npz_path, f"class_{label}")
    os.makedirs(class_dir, exist_ok=True)
    save_path = os.path.join(class_dir, f"{stem}_label{label}_seqfeat.npz")

    if os.path.exists(save_path):
        print(f"⚙️ 跳过已存在 IMF-token: {save_path}")
        return None

    # 加载 VMD 结果
    vmd = np.load(f_path, allow_pickle=True)
    seg_results = vmd["seg_vmd_results"].tolist()
    fs = int(vmd.get("fs", CFG.seg.fs_target))

    print(f"\n=== 并行处理文件: {stem}, label={label}, seg={len(seg_results)} ===")

    seg_tokens_list = []
    seg_fcenters_list = []

    for seg_idx, seg_res in enumerate(seg_results):
        U = seg_res["U_fix"]
        e = seg_res["e_fix"]

        # -------- IMF 筛选 --------
        if getattr(CFG.post, "use_fbe", False):
            U_sel, _ = select_imfs_by_FBE(
                U, fs,
                use_envelope=True,
                top_k=CFG.post.fbe_top_k
            )
        else:
            U_sel = prune_merge_imfs(
                U,
                CFG.post.prune_energy_thr,
                CFG.post.prune_corr_thr
            )

        if U_sel.size == 0:
            continue

        # -------- 加权重构 --------
        w = compute_weights(U_sel, e, CFG.post.weight_eta)
        w = np.maximum(w, 1e-6)
        w /= w.sum()

        s_hat, U_den = reconstruct(U_sel, e, w, CFG.post.wavelet, CFG.post.wavelet_levels)
        feats, feat_names = IMF_features_list(U_den, fs)

        # -------- 中心频率 --------
        f_center = []
        for k in range(U_den.shape[0]):
            Uk = np.fft.rfft(U_den[k])
            freqs = np.fft.rfftfreq(len(U_den[k]), 1 / fs)
            P = np.abs(Uk) ** 2 + 1e-12
            f_center.append((freqs * P).sum() / P.sum())
        f_center = np.array(f_center, dtype=float)

        seg_tokens_list.append(np.asarray(feats, dtype=float)) # shape (K_seg, D)
        seg_fcenters_list.append(f_center)

    if len(seg_tokens_list) == 0:
        print(f"⚠ 文件 {stem} 无有效 IMF token")
        return None

    np.savez(
        save_path,
        seg_tokens=np.array(seg_tokens_list, dtype=object),
        seg_fcenters=np.array(seg_fcenters_list, dtype=object),
        label=label,
        filename=stem
    )

    print(f"📦 saved: {save_path}")
    return save_path


from joblib import Parallel, delayed
import multiprocessing


def step2_imf_token_representation(
        vmd_save_dir: str,
        IMF_npz_path: str,
        n_jobs: int = None
):
    os.makedirs(IMF_npz_path, exist_ok=True)

    vmd_files = [f for f in os.listdir(vmd_save_dir) if f.endswith("_vmd.npz")]
    assert len(vmd_files) > 0, f"未找到 VMD 文件在目录 {vmd_save_dir}"

    print(f"共加载 {len(vmd_files)} 个 VMD 文件")

    if n_jobs is None:
        n_jobs = max(1, multiprocessing.cpu_count() - 1)

    print(f"🚀 使用 {n_jobs} 个并行进程")

    results = Parallel(
        n_jobs=n_jobs,
        backend="loky",
        verbose=10
    )(
        delayed(process_single_vmd_file)(
            f, vmd_save_dir, IMF_npz_path
        )
        for f in vmd_files
    )

    results = [r for r in results if r is not None]
    print(f"\n### Step2 完成，共生成 {len(results)} 个 token 文件 ###")


"""
每一种特征组（如 Mel 12维）整体作为一个 Token。

token 数 = 模态数，不依赖 top_k。

Random Forest 不允许 token 序列，但允许融合后的向量，因此后面要把 token 融合成一个向量。

"""