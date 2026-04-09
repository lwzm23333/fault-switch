"""
plot_vmd_result.py
用于可视化已保存的VMD分解结果（从npz文件读取）
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D


# ============================================================
# 1️⃣ 主可视化函数：读取VMD结果并绘制信号及IMF
# ============================================================
def plot_vmd_npz(npz_path: str, save_dir: str = "plots", max_imf_to_show: int = 6):
    os.makedirs(save_dir, exist_ok=True)
    data = np.load(npz_path, allow_pickle=True)

    stem = Path(npz_path).stem
    print(f"\n🎯 文件: {Path(npz_path).name}")
    print(f"包含字段: {list(data.keys())}")

    # ----------- 安全读取各字段 -----------
    x = data.get("x_prep", None)
    x_recon = data.get("x_recon", None)
    e_full = data.get("e_full", None)
    U_full = data.get("U_full", None)
    metrics = data.get("metrics", None)
    alpha_list = data.get("alpha_list", None)
    K_list = data.get("K_list", None)
    global_centers = data.get("global_centers", None)

    # ----------- 打印基本信息 -----------
    if U_full is not None:
        print(f"信号长度: {U_full.shape[1]}, IMF数: {U_full.shape[0]}")
    if metrics is not None:
        m = metrics.item() if isinstance(metrics, np.ndarray) else metrics
        print(f"质量指标: "
              f"RMSE={m.get('recon_rmse', 0):.4f}, "
              f"残差比={m.get('residual_energy_ratio', 0):.4f}, "
              f"谱熵差={m.get('spectral_entropy_diff', 0):.4f}")

    # =====================================================
    # 绘制信号与重构
    # =====================================================
    plt.figure(figsize=(12, 8))

    if x is not None:
        plt.subplot(3, 1, 1)
        plt.title("原始信号")
        plt.plot(x, color='gray')
    if x_recon is not None:
        plt.subplot(3, 1, 2)
        plt.title("VMD 重构信号")
        plt.plot(x_recon, color='blue')
    if e_full is not None:
        plt.subplot(3, 1, 3)
        plt.title("残差信号")
        plt.plot(e_full, color='red')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"{stem}_overview.png"), dpi=300)
    plt.close()
    print(f"✅ 保存基础可视化: {os.path.join(save_dir, f'{stem}_overview.png')}")

    # =====================================================
    # 绘制部分 IMF
    # =====================================================
    if U_full is not None:
        K = U_full.shape[0]
        n_show = min(K, max_imf_to_show)
        plt.figure(figsize=(12, 2 * n_show))
        for i in range(n_show):
            plt.subplot(n_show, 1, i + 1)
            plt.plot(U_full[i], lw=0.8)
            plt.title(f"IMF {i + 1}")
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"{stem}_imfs.png"), dpi=300)
        plt.close()
        print(f"✅ 保存 IMF 分布图: {os.path.join(save_dir, f'{stem}_imfs.png')}")

        # 绘制 3D IMF 结构
        plot_imfs_3d(U_full, f"IMFs - {stem}",
                     save_path=os.path.join(save_dir, f"{stem}_imf3d.png"))


    # =====================================================
    # 绘制 α-K 分布趋势
    # =====================================================
    if alpha_list is not None and K_list is not None:
        plot_alpha_K(alpha_list, K_list, stem, save_dir)
    else:
        print("⚠️ 文件中未检测到 α 或 K 数据，跳过趋势绘制。")

    # =====================================================
    # 绘制频带对齐结果
    # =====================================================
    if global_centers is not None:
        plot_global_freq_lock(global_centers, stem, save_dir)
    else:
        print("⚠️ 文件中未检测到 'global_centers'，跳过频带可视化。")


# ============================================================
# 2️⃣ 三维 IMF 绘制
# ============================================================
def plot_imfs_3d(U, title, save_path=None):
    """
    绘制三维IMF结构图
    U: np.ndarray, shape=(K, N)
    """
    K, N = U.shape
    t = np.arange(N)
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection="3d")

    for k in range(K):
        ax.plot(t, np.full_like(t, k), U[k, :], lw=1.2)

    ax.set_xlabel("Time Index")
    ax.set_ylabel("IMF Index")
    ax.set_zlabel("Amplitude")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ 保存3D IMF图: {save_path}")
    else:
        plt.show()
    plt.close()


# ============================================================
# 3️⃣ α-K 分布趋势
# ============================================================
def plot_alpha_K(alpha_list, K_list, stem, save_dir):
    plt.figure(figsize=(8, 4))
    plt.plot(alpha_list, label="α (bandwidth penalty)", color='tab:blue')
    plt.plot(K_list, label="K (number of modes)", color='tab:orange')
    plt.legend()
    plt.title(f"Frame-wise α and K Distribution - {stem}")
    plt.xlabel("Frame Index")
    plt.ylabel("Value")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{stem}_alpha_K.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存 α-K 分布图: {save_path}")


# ============================================================
# 4️⃣ 频带对齐结果绘制
# ============================================================
def plot_global_freq_lock(global_centers, wav_stem, save_dir):
    plt.figure(figsize=(6, 4))
    plt.plot(global_centers, np.arange(len(global_centers)), "o-", color="tab:green")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("IMF Index")
    plt.title(f"Global Frequency Lock - {wav_stem}")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{wav_stem}_freq_lock.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"✅ 保存频带对齐图: {save_path}")


# ============================================================
# 5️⃣ 主入口
# ============================================================
if __name__ == "__main__":
    npz_path = r"outputs/imfs_5/filenamez10_label0_vmd.npz"  # ← 修改为目标文件路径
    plot_vmd_npz(npz_path, save_dir="outputs/plots")
