from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from torch import cdist
def fuzzy_rough_dependency_degree(X, y, sigma=1.0):
    """
    FRDD：使用 NumPy 计算高斯相似度，避免 torch/scipy 的 cdist 冲突
    """
    N, D = X.shape
    score = np.zeros(D)

    for d in range(D):
        xd = X[:, d]

        # 使用 NumPy 手动构建距离矩阵（无外部依赖）
        diff = xd[:, None] - xd[None, :]
        mu = np.exp(-(diff ** 2) / (2 * sigma * sigma))  # 高斯模糊相似度 (N × N)

        # 类标一致性
        label_sim = (y[:, None] == y[None, :]).astype(float)

        # fuzzy positive region
        gamma = np.min(np.maximum(mu, label_sim), axis=1).mean()
        score[d] = gamma

    return score


def step3_feature_selection_fuzzy(
        feat_npz_path: str,
        selected_npz_path: str,
        top_k: int = 30
):
    print("\n===== Step3: Feature Selection =====")

    # ---------- 1. 加载特征 ----------
    data = np.load(feat_npz_path, allow_pickle=True)
    X = data["X"].astype(float)
    y = data["y"]

    N, D = X.shape
    print(f"加载特征矩阵 X: {X.shape}, 标签 y: {y.shape}")

    # =====================================================
    # 2. 删除零方差特征（防止互信息报错）
    # =====================================================
    selector = VarianceThreshold(threshold=1e-8)
    X_var = selector.fit_transform(X)
    valid_idx = selector.get_support(indices=True)

    print(f"删除零方差特征后 X_var shape = {X_var.shape}（原 {D} -> {len(valid_idx)}）")

    # =====================================================
    # 3. 计算特征重要度
    # =====================================================

    # ----- (1) 方差 -----
    var_score = X.var(axis=0)
    var_score = var_score[valid_idx]
    var_score = MinMaxScaler().fit_transform(var_score.reshape(-1, 1)).ravel()

    # ----- (2) 互信息（安全版） -----
    try:
        mi_score = mutual_info_classif(X_var, y, n_neighbors=2, random_state=0)
    except:
        print("⚠️ MI 计算失败，尝试 n_neighbors=1 ...")
        try:
            mi_score = mutual_info_classif(X_var, y, n_neighbors=1, random_state=0)
        except:
            print("❌ MI 仍失败，使用全零评分！")
            mi_score = np.zeros(X_var.shape[1])
    mi_score = MinMaxScaler().fit_transform(mi_score.reshape(-1, 1)).ravel()
    print(f"MI shape: {mi_score.shape}")

    # ----- (3) RF重要度 -----
    rf = RandomForestClassifier(n_estimators=300, random_state=0)
    rf.fit(X_var, y)
    rf_score = rf.feature_importances_
    rf_score = MinMaxScaler().fit_transform(rf_score.reshape(-1, 1)).ravel()

    # ----- (4) FRDD -----
    frdd_score = fuzzy_rough_dependency_degree(X_var, y)
    frdd_score = MinMaxScaler().fit_transform(frdd_score.reshape(-1, 1)).ravel()

    # ----- (5) 皮尔逊相关性 -----
    corr_list = []
    for i in range(X_var.shape[1]):
        if np.std(X_var[:, i]) < 1e-8:
            corr_list.append(0.0)
        else:
            corr = np.corrcoef(X_var[:, i], y)[0, 1]
            corr_list.append(abs(corr))  # 绝对值
    corr_score = MinMaxScaler().fit_transform(np.array(corr_list).reshape(-1, 1)).ravel()

    print("\n=== 各类特征分数统计 ===")

    def show_stat(name, arr):
        print(f"{name:12s} | min={arr.min():.4f} max={arr.max():.4f} mean={arr.mean():.4f} std={arr.std():.4f}")

    show_stat("Variance", var_score)
    show_stat("MutualInfo", mi_score)
    show_stat("RF", rf_score)
    show_stat("FRDD", frdd_score)
    show_stat("Correlation", corr_score)

    # =====================================================
    # 4. 加权融合（加入相关性）
    # =====================================================
    from scipy.stats import spearmanr

    scores = np.vstack([var_score, mi_score, rf_score, frdd_score, corr_score])  # shape = (5, n_features)

    R = np.zeros((5, 5))
    for i in range(5):
        for j in range(5):
            R[i, j], _ = spearmanr(scores[i], scores[j])


    reliability = np.mean(np.abs(R), axis=1)  # or use method3's reliability
    weights = reliability / reliability.sum()
    w_var = weights[0]
    w_mi = weights[1]
    w_rf = weights[2]
    w_fr = weights[3]
    w_corr = weights[4]
    print(w_var, w_mi, w_rf, w_fr, w_corr)

    F_score = (w_var * var_score +
               w_mi * mi_score +
               w_rf * rf_score +
               w_fr * frdd_score +
               w_corr * corr_score)

    # =====================================================
    # 5. 选择 top-K
    # =====================================================
    idx_sorted = np.argsort(F_score)[::-1]
    idx_top = idx_sorted[:top_k]

    # 映射回原始特征编号
    final_idx = valid_idx[idx_top]
    X_sel = X[:, final_idx]

    # 保存
    np.savez(selected_npz_path, X=X_sel, y=y, feat_idx=final_idx)

    print(f"\n🎉 Step3 完成：选择 {top_k} 个特征 (原特征编号: {final_idx})")
    print(f"保存至: {selected_npz_path}")

    import matplotlib.pyplot as plt
    import seaborn as sns

    # ===========================================
    # 1. 打印 Top-K 特征表
    # ===========================================
    print("\n================ Top-K 特征表 ================\n")
    print(f"{'Rank':<5}{'FeatureID':<12}{'RawID':<12}{'F_score':<10}")
    print("-" * 45)
    for r, idx in enumerate(idx_top):
        print(f"{r + 1:<5}{idx:<12}{final_idx[r]:<12}{F_score[idx]:<10.4f}")

    # ===========================================
    # 2. Top-K F-score 柱状图
    # ===========================================
    plt.figure(figsize=(10, 5))
    plt.bar(np.arange(top_k), F_score[idx_top])
    plt.xticks(np.arange(top_k), final_idx, rotation=45)
    plt.ylabel("F-score")
    plt.title(f"Top-{top_k} Feature Importance")
    plt.tight_layout()
    plt.savefig("topk_fscore.png", dpi=300)
    plt.show()

    # ===========================================
    # 3. 雷达图 (5 scoring types)
    # ===========================================

    import math

    labels = ["Var", "MI", "RF", "FRDD", "Corr"]
    score_means = [
        var_score.mean(),
        mi_score.mean(),
        rf_score.mean(),
        frdd_score.mean(),
        corr_score.mean()
    ]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    score_means += score_means[:1]
    angles += angles[:1]

    plt.figure(figsize=(6, 6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, score_means, linewidth=2)
    ax.fill(angles, score_means, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Scores Radar Chart")
    plt.savefig("scores_radar.png", dpi=300)
    plt.show()

    # ===========================================
    # 4. 热力图 (feature × scoring)
    # ===========================================

    all_scores = np.vstack([var_score, mi_score, rf_score, frdd_score, corr_score])
    plt.figure(figsize=(12, 4))
    sns.heatmap(all_scores, cmap="viridis", cbar=True, yticklabels=labels)
    plt.title("Feature Scores Heatmap")
    plt.xlabel("Feature Index")
    plt.tight_layout()
    plt.savefig("scores_heatmap.png", dpi=300)
    plt.show()

    return final_idx, F_score


