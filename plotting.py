# -*- coding: utf-8 -*-
"""
visualize_results_modular.py
功能：
1) analyze_all_features(): 对整个文件夹进行 PCA / t-SNE 聚类分析；
2) visualize_single_sample(): 对指定 npz 文件进行详细可视化；
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# ================================================================
# 通用配置
# ================================================================
FEATURE_DIR = "outputs/fused_features/test_clean"     # 输入 npz 文件夹
SAVE_DIR = "outputs/figs_visual"    # 输出图片保存路径
os.makedirs(SAVE_DIR, exist_ok=True)


import matplotlib.pyplot as plt
import seaborn as sns
# ================================================================
# (1) 全局特征分析函数：PCA / t-SNE
# ================================================================
def analyze_all_features(feature_dir, save_dir):
    """对整个目录下的所有 npz 文件执行 PCA / t-SNE 聚类分析"""
    files = sorted(Path(feature_dir).rglob("*.npz"))
    assert files, f"未在 {feature_dir} 中发现 npz 文件。"
    print(f"发现 {len(files)} 个 npz 文件")

    X_list, y_list = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        X_list.append(d["fused_features"])
        y_list.append(int(d["labels"]))
    X = np.vstack(X_list)
    y = np.array(y_list)

    def plot_embedding(X_embedded, y, method, out_path):
        plt.figure(figsize=(8, 6))
        scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1],
                              c=y, cmap="tab10", s=25, alpha=0.8)
        plt.legend(*scatter.legend_elements(), title="Label", loc="best")
        plt.title(f"{method} Visualization of Z_file Features")
        plt.xlabel(f"{method}-1")
        plt.ylabel(f"{method}-2")
        plt.tight_layout()
        plt.savefig(out_path, dpi=180)
        plt.close()
        print(f"✅ 已保存 {method} 图: {out_path}")

    # ---- PCA ----
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plot_embedding(X_pca, y, "PCA", os.path.join(save_dir, "pca_test_2.png"))

    # ---- t-SNE ----
    X_tsne = TSNE(n_components=2, random_state=42, init="pca", metric="cosine", n_iter=2000,
                  perplexity=10, learning_rate=200).fit_transform(X)
    plot_embedding(X_tsne, y, "t-SNE", os.path.join(save_dir, "tsne_test_2.png"))

    print("✅ 全局特征分析完成。")

def plot_alpha_K(alpha_list, K_list, segment_rmse=None):
    # 1️⃣ alpha 迭代曲线
    plt.figure(figsize=(8,4))
    plt.plot(alpha_list, marker='o')
    plt.xlabel("Iteration")
    plt.ylabel(r"$\alpha$")
    plt.title("Adaptive α over Iterations")
    plt.grid(True)
    plt.show()

    # 2️⃣ K 分布直方图
    plt.figure(figsize=(6,4))
    sns.histplot(K_list, bins=range(min(K_list), max(K_list)+2), kde=False)
    plt.xlabel("Recommended K per Segment")
    plt.ylabel("Count")
    plt.title("Distribution of Adaptive K")
    plt.show()

    # 3️⃣ α vs K 散点图，颜色映射 RMSE（可选）
    plt.figure(figsize=(6,5))
    if segment_rmse is not None:
        plt.scatter(K_list, alpha_list, c=segment_rmse, cmap="viridis", s=50)
        plt.colorbar(label="Segment RMSE")
    else:
        plt.scatter(K_list, alpha_list, c='blue', s=50)
    plt.xlabel("Adaptive K")
    plt.ylabel("Adaptive α")
    plt.title("Adaptive K vs α across Segments")
    plt.grid(True)
    plt.show()


# ================================================================
# IMF 3D 绘制函数
# ================================================================
def plot_imfs_3d(imfs, title, out_png, spacing=2.0):
    """绘制 IMF 三维曲线图"""
    from mpl_toolkits.mplot3d import Axes3D  # noqa
    K, N = imfs.shape
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")
    xs = np.arange(N)
    for k in range(K):
        ys = np.full(N, (k + 1) * spacing)
        ax.plot(xs, ys, imfs[k], linewidth=0.6, alpha=0.8, label=f"IMF {k+1}")
    ax.set_xlabel("Sampling Points")
    ax.set_ylabel("Mode Index")
    ax.set_zlabel("Amplitude")
    ax.set_title(title)
    ax.legend(loc="upper left", bbox_to_anchor=(1.05, 1.0))
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"✅ 已保存 IMF 3D 图: {out_png}")



# ================================================================
# (2) 单文件可视化函数
# ================================================================
def visualize_single_sample(npz_path, save_dir):
    """对单个 npz 文件绘制融合特征、注意力、信号对比、IMF 三维图"""
    data = np.load(npz_path, allow_pickle=True)
    print(f"\n🎯 目标文件: {npz_path}")
    print("包含键:", list(data.keys()))

    Z_file = data.get("Z_file")
    label = int(data.get("label", -1))
    s_hat_full = data.get("s_hat_full")
    attn_maps = data.get("attn_maps")
    print(attn_maps.shape)
    U_den = data.get("U_den")
    print(U_den.shape)
    x_raw = data.get("x_raw")

    stem = Path(npz_path).stem

    # === (1) IMF-token 融合特征可视化 ===
    if Z_file is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(Z_file, marker="o", linewidth=1.2)
        plt.title(f"Feature vector (label={label}) - {stem}")
        plt.xlabel("Feature index")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_features.png"), dpi=180)
        plt.close()

    # === (2) 注意力权重热力图 ===
    if attn_maps is not None and len(attn_maps) > 0:
        try:
            # 计算每帧 IMF 数
            max_len = max(len(a) for a in attn_maps)
            # 填充到相同长度（右侧填充 0）
            attn_padded = np.zeros((len(attn_maps), max_len))
            for i, a in enumerate(attn_maps):
                attn_padded[i, :len(a)] = a  # 右侧补0

            # 求平均注意力（帧间平均）
            attn_mean = np.mean(attn_padded, axis=0)

            plt.figure(figsize=(8, 4))
            sns.heatmap(attn_mean[None, :], cmap="YlGnBu", cbar=True,
                        xticklabels=False, yticklabels=["attention"])
            plt.title(f"Attention map (mean across frames, padded) - label={label}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{stem}_attn.png"), dpi=180)
            plt.close()

            print(f"✅ 注意力权重热力图绘制完成，IMF最大长度={max_len}")
        except Exception as e:
            print(f"⚠️ 注意力权重绘制失败: {e}")
    else:
        print("⚠️ 未检测到 attn_maps。")

    # === (3) IMF 三维绘制 ===
    if "U_den" in data:
        U_den = data["U_den"]
        if isinstance(U_den, np.ndarray) and U_den.ndim == 2:
            plot_imfs_3d(U_den, f"IMFs - {stem}",
                         os.path.join(save_dir, f"{stem}_imf3d.png"))
        else:
            print(f"⚠️ U_den 结构异常: type={type(U_den)}, shape={getattr(U_den, 'shape', None)}")
    else:
        print("⚠️ 该文件中未检测到 'U_den'。")

    # === (4) 原始信号与去噪信号对比 ===
    if x_raw is not None and s_hat_full is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(x_raw, linewidth=0.8)
        plt.title("Original signal")
        plt.subplot(2, 1, 2)
        plt.plot(s_hat_full, linewidth=0.8)
        plt.title("Denoised signal (VMD reconstructed)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_compare_signal.png"), dpi=180)
        plt.close()
    elif s_hat_full is not None:
        plt.figure(figsize=(10, 3))
        plt.plot(s_hat_full, linewidth=0.8)
        plt.title(f"Denoised signal - label={label}")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_signal.png"), dpi=180)
        plt.close()
    else:
        print("⚠️ 未检测到 's_hat_full' 或 'x_raw'。")

    print(f"✅ 单文件可视化完成: {stem}")

def vmd_single_sample(npz_path, save_dir):
    """对单个 npz 文件绘制融合特征、注意力、信号对比、IMF 三维图"""
    data = np.load(npz_path, allow_pickle=True)
    print(f"\n🎯 目标文件: {npz_path}")
    print("包含键:", list(data.keys()))

    label = int(data.get("label", -1))
    seg_vmd_results = data.get("seg_vmd_results")
    print(seg_vmd_results.shape)
    concatenated_U = data.get("concatenated_U")
    x_raw = data.get("x_raw")

    stem = Path(npz_path).stem

    # 在 vmd_single_sample 函数中修改IMF绘图部分
    if "seg_vmd_results" in data:
        seg_vmd = data["seg_vmd_results"]

        # 处理对象数组
        if isinstance(seg_vmd, np.ndarray) and seg_vmd.dtype == object:
            # 提取第一段的IMF进行展示
            first_segment = seg_vmd[0]
            if isinstance(first_segment, dict) and "U_fix" in first_segment:
                imfs = first_segment["U_fix"]
                if imfs.ndim == 2:
                    plot_imfs_3d(imfs, f"IMFs - {stem} (First Segment)",
                                 os.path.join(save_dir, f"{stem}_imf3d.png"))
                else:
                    print(f"⚠️ IMF数据维度异常: {imfs.shape}")
            else:
                print("⚠️ 段数据结构异常")
        elif isinstance(seg_vmd, list):
            # 如果是列表格式
            first_segment = seg_vmd[0]
            if isinstance(first_segment, dict) and "U_fix" in first_segment:
                imfs = first_segment["U_fix"]
                if imfs.ndim == 2:
                    plot_imfs_3d(imfs, f"IMFs - {stem} (First Segment)",
                                 os.path.join(save_dir, f"{stem}_imf3d.png"))

    # === (2) 原始信号与去噪信号对比 ===
    if x_raw is not None and concatenated_U is not None:
        plt.figure(figsize=(10, 4))
        plt.subplot(2, 1, 1)
        plt.plot(x_raw, linewidth=0.8)
        plt.title("Original signal")
        plt.subplot(2, 1, 2)
        plt.plot(concatenated_U, linewidth=0.8)
        plt.title("Denoised signal (VMD reconstructed)")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_compare_signal.png"), dpi=180)
        plt.close()
    elif concatenated_U is not None:
        plt.figure(figsize=(10, 3))
        plt.plot(concatenated_U, linewidth=0.8)
        plt.title(f"Denoised signal - label={label}")
        plt.xlabel("Sample index")
        plt.ylabel("Amplitude")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_signal.png"), dpi=180)
        plt.close()
    else:
        print("⚠️ 未检测到 'U_den' 或 'x'。")

    print(f"✅ 单文件可视化完成: {stem}")

def features_sample(npz_path, save_dir):
    """对单个 npz 文件绘制融合特征、注意力、信号对比、IMF 三维图"""
    if not os.path.exists(npz_path):
        print(f"❌ 文件不存在: {npz_path}")
        return

    data = np.load(npz_path, allow_pickle=True)
    print(f"\n🎯 目标文件: {npz_path}")
    print("包含键:", list(data.keys()))

    file_feat = data.get("file_feat")  # 注意：键名改为 file_feat
    label = int(data.get("label", -1))
    seg_attn = data.get("seg_attn")  # 注意：键名改为 seg_attn
    U_den = data.get("full_reconstructed")
    full_imf_matrix = data.get("full_imf_matrix")

    stem = Path(npz_path).stem
    os.makedirs(save_dir, exist_ok=True)

    # === IMF 三维图 ===
    if "full_imf_matrix" in data:
        full_imf_matrix = data["full_imf_matrix"]
        if isinstance(full_imf_matrix, np.ndarray) and full_imf_matrix.ndim == 2:
            plot_imfs_3d(full_imf_matrix, f"IMFs - {stem}",
                         os.path.join(save_dir, f"{stem}_imf3d.png"))
        else:
            print(f"⚠️ IMFS 结构异常: type={type(full_imf_matrix)}, shape={getattr(full_imf_matrix, 'shape', None)}")
    else:
        print("⚠️ 该文件中未检测到 'full_imf_matrix'。")

    # === 特征向量可视化 ===
    if file_feat is not None:
        plt.figure(figsize=(10, 4))
        plt.plot(file_feat, marker="o", linewidth=1.2)
        plt.title(f"Feature vector (label={label}) - {stem}")
        plt.xlabel("Feature index")
        plt.ylabel("Value")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_features.png"), dpi=180)
        plt.close()

    # === 注意力权重热力图 ===
    if seg_attn is not None and len(seg_attn) > 0:
        try:
            # 注意：seg_attn 现在已经是numpy数组
            attn_list = seg_attn.tolist() if hasattr(seg_attn, 'tolist') else seg_attn

            max_len = max(len(a) for a in attn_list)
            attn_padded = np.zeros((len(attn_list), max_len))
            for i, a in enumerate(attn_list):
                attn_padded[i, :len(a)] = a

            attn_mean = np.mean(attn_padded, axis=0)

            plt.figure(figsize=(8, 4))
            sns.heatmap(attn_mean[None, :], cmap="YlGnBu", cbar=True,
                        xticklabels=False, yticklabels=["attention"])
            plt.title(f"Attention map (mean across frames) - label={label}")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f"{stem}_attn.png"), dpi=180)
            plt.close()

            print(f"✅ 注意力权重热力图绘制完成，IMF最大长度={max_len}")
        except Exception as e:
            print(f"⚠️ 注意力权重绘制失败: {e}")
    else:
        print("⚠️ 未检测到 seg_attn。")

import matplotlib.pyplot as plt

#  可视化  α、K 分布
# data = np.load("outputs/imfs/sample_label1_vmd.npz", allow_pickle=True)
# alpha_list = data["alpha_list"]
# K_list = data["K_list"]
#
# plt.figure(figsize=(8,4))
# plt.plot(alpha_list, label="α")
# plt.plot(K_list, label="K")
# plt.legend()
# plt.title("每帧 α 与 K 分布")
# plt.xlabel("Frame Index")
# plt.show()

# 可视化各文件/段之间的频带对齐效果，
# plt.figure()
# plt.plot(global_centers, np.arange(len(global_centers)), 'o-')
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('IMF index')
# plt.title(f'Global Frequency Lock - {wav_stem}')
# plt.grid(True)

# ================================================================
# 主函数调用
# ================================================================
if __name__ == "__main__":
    # 1) 对整个文件夹做 t-SNE 和 PCA 聚类
    analyze_all_features(FEATURE_DIR, SAVE_DIR)

    # # 2) 随机抽取一个样本进行详细可视化
    # all_files = sorted(Path(FEATURE_DIR).rglob("*.npz"))
    # if all_files:
    #     random_file = random.choice(all_files)
    #     #visualize_single_sample("outputs/files_2/filenamez10_label0.npz", SAVE_DIR)     # 分开保存之前的查看
    #     #vmd_single_sample("outputs/files_2/filenamez10_label1.npz", SAVE_DIR)
    #     features_sample("outputs/train_feat/class_3/filenamez2_label3_feat.npz", SAVE_DIR)
    # else:
    #     print("未发现 npz 文件。")
