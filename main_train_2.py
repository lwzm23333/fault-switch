import os
import shutil
import pickle
import numpy as np
from pathlib import Path
import random
import json
from collections import defaultdict
from dataset import split_dataset
from config import CFG
from sklearn.preprocessing import StandardScaler
from adaptive_vmd import step1_vmd_decompose
from extract_features import step2_extract_features_plain
from features_select import step3_feature_selection_fuzzy
from extract_features_token import step2_imf_token_representation
from features_fuser import step3_fuse_tokens
from train_classifier import build_model, crossval_train_eval

# --------------------------------------------------
# 数据集构造
# --------------------------------------------------
# ===== 1.音频加载函数 =====
def load_dataset(root):
    wav_files = []
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    for lbl, cname in enumerate(class_names):
        folder = os.path.join(root, cname)
        for fn in os.listdir(folder):
            if fn.lower().endswith(".wav"):
                wav_files.append((os.path.join(folder, fn), lbl))
    return wav_files, class_names

# ===== 2.划分数据集 =====
def prepare_data(data_root, force=False):
    """
    划分 train/test 并把文件复制到 data_root/Train 和 data_root/Test。
    若已存在并且 force=False，则直接返回已存在路径。 """
    train_dir = os.path.join(data_root, "train")
    test_dir = os.path.join(data_root, "test")
    if os.path.exists(train_dir) and os.path.exists(test_dir) and not force:
        print(f"[prepare_data] train/test already exist: {train_dir}, {test_dir}")
        return train_dir, test_dir
    wav_files, class_names = load_dataset(data_root)
    train_list, test_list = split_dataset(wav_files, test_ratio=0.375, seed=42, save_path="data/split.json")
    # 清空并复制（如果 force=True，可覆盖）
    for root_dir in [train_dir, test_dir]:
        os.makedirs(root_dir, exist_ok=True)

    for dataset, root_dir in [(train_list, train_dir), (test_list, test_dir)]:
        for filepath, label in dataset:
            class_name = class_names[label]
            dst_dir = os.path.join(root_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, os.path.basename(filepath))
            if not os.path.exists(dst_file):
                shutil.copy(filepath, dst_file)
    print(f"[prepare_data] Data copied to: {train_dir} and {test_dir}")
    return train_dir, test_dir


# --------------------------------------------------
# 数据处理
# --------------------------------------------------
# ===== 1.特征选择 =====
def feature_select(
        vmd_save_dir: str,
        feat_excel_path: str,
        feat_npz_path: str,
        all_npz_path:str,
        selected_npz_path: str,
        top_k: int = 30
):
    """
    读取训练集的wav文件，完成VMD分解 → 特征提取 → 特征筛选
    :param wav: 训练集数据
    :param fs: 额定频率
    输出的特征作为后续IMF-token进行特征提取的参考
    """
    # step2 对IMF组的特征提取             extract_features.py
    step2_extract_features_plain(vmd_save_dir, feat_excel_path, feat_npz_path)

    # step3 读取excel特征，进行特征筛选     features_select.py
    step3_feature_selection_fuzzy(all_npz_path, selected_npz_path, top_k)

# ===== 2.特征加载 =====
def load_features(feature_dir):
    """
    统一加载 fused feature 目录下每个 class_*/*.npz
    要求每个 npz 至少包含键 'fused_features' 和 'labels'
    """
    X_list, y_list = [], []
    class_dirs = sorted(Path(feature_dir).glob("class_*"))
    for class_dir in class_dirs:
        for npz_file in class_dir.glob("*.npz"):
            d = np.load(npz_file, allow_pickle=True)
            # <-- 注意：期望 'fused_features' 和 'labels' 这两个 key 存在
            if "fused_features" not in d or "labels" not in d:
                raise RuntimeError(f"Expected keys not found in {npz_file}. Keys: {list(d.keys())}")
            X_list.append(d["fused_features"])
            y_list.append(int(d["labels"]))
    if len(X_list) == 0:
        raise RuntimeError(f"No fused features found in {feature_dir}")
    X = np.vstack(X_list)
    y = np.array(y_list)
    print(f"[load_features] Loaded {len(y)} samples from {feature_dir}. X.shape={X.shape}")
    return X, y

def train_model(X_train, y_train, model_name="rf"):
    clf = build_model(model_name)
    clf.fit(X_train, y_train)
    return clf

def main_train():
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"

    # ========= 1. 数据准备 =========
    train_dir, _ = prepare_data(data_root)

    # ========= 2. 特征提取（Train only） =========
    vmd_train = "outputs/vmd_results/train_vmd"
    imf_train = "outputs/imf_tokens/train"
    fused_train = "outputs/fused_features/train"

    # ========= 3.筛选部分 =======
    feat_all_excel_path = "outputs/feat_all_excel.xlsx"
    feat_npz_path = "outputs/train_all_feat"
    all_npz_path = "outputs/train_all_feat/all_file_features.npz"
    selected_npz_path = "outputs/train_all_feat/selected_features.npz"
    top_k = 30

    step1_vmd_decompose(train_dir, vmd_train)
    # feature_select(vmd_train, feat_all_excel_path, feat_npz_path, all_npz_path, selected_npz_path, top_k)
    step2_imf_token_representation(vmd_train, imf_train)
    step3_fuse_tokens(CFG, imf_train, fused_train)


    # ========= 3. 加载训练特征 =========
    X_train, y_train = load_features(fused_train)
    print("Train std:", X_train.std(axis=0).mean())


    # ========= 3.5 交叉验证评估（仅用于分析，不保存模型） =========
    cv_results = crossval_train_eval(
        X_train,
        y_train,
        model_name="rf",
        n_splits=5,
        return_preds=False
    )

    print("\n[Cross-validation results]")
    for k, v in cv_results.items():
        print(f"{k}: {v:.4f}")

    # ========= 4. 模型训练 =========
    clf = train_model(
        X_train, y_train,
        model_name="rf")

    # ========= 5. 保存模型 =========
    os.makedirs("outputs/models", exist_ok=True)
    model_path = "outputs/models/rf_fuser.pkl"
    with open(model_path, "wb") as f:
        import pickle
        pickle.dump(clf, f)

    print(f"✅ 训练完成，模型已保存至 {model_path}")

# --------------------------------------------------
# Demo 主程序（构造示例数据）
# --------------------------------------------------
if __name__ == "__main__":
    main_train()