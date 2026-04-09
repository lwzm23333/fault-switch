import os
import shutil
import numpy as np
from config import CFG
from pathlib import Path
from sklearn.svm import SVC
from dataset import split_dataset
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# 假设以下函数已实现并可用（根据您的项目结构修改导入路径）
from adaptive_vmd import step1_vmd_decompose
from extract_features import step2_extract_features_plain
from features_select import step3_feature_selection_fuzzy
from extract_features_token import step2_imf_token_representation
from features_fuser import step3_fuse_tokens
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from train_classifier import crossval_train_eval, build_model
from noise_data import make_noisy_testset

def load_dataset(root):
    wav_files = []
    class_names = sorted(os.listdir(root))

    for lbl, cname in enumerate(class_names):
        folder = os.path.join(root, cname)
        if not os.path.isdir(folder):
            continue

        for fn in os.listdir(folder):
            if fn.lower().endswith(".wav"):
                wav_files.append((os.path.join(folder, fn), lbl))

    return wav_files, class_names

def prepare_data(data_root):
    """
    划分数据集并复制文件到 Train/Test 文件夹中。
    """
    wav_files, class_names = load_dataset(data_root)
    # 划分训练/测试集（70% 训练，30% 测试），保存划分结果到 JSON
    train_list, test_list = split_dataset(wav_files, test_ratio=0.3, seed=42, save_path="data/split.json")

    # 创建训练/测试文件夹并复制音频
    train_dir = os.path.join(data_root, "Train")
    test_dir = os.path.join(data_root, "Test")
    for dataset, root_dir in [(train_list, train_dir), (test_list, test_dir)]:
        for filepath, label in dataset:
            class_name = class_names[label]
            dst_dir = os.path.join(root_dir, class_name)
            os.makedirs(dst_dir, exist_ok=True)
            dst_file = os.path.join(dst_dir, os.path.basename(filepath))
            shutil.copy(filepath, dst_file)
    print(f"数据已复制到: {train_dir} 和 {test_dir}")
    return train_dir, test_dir

def extract_and_fuse_features(train_dir, test_dir, vmd_dir, imf_dir, fused_dir):
    """
    对训练集和测试集分别执行VMD分解、特征提取及融合。
    """

    os.makedirs(vmd_dir, exist_ok=True)
    os.makedirs(imf_dir, exist_ok=True)
    os.makedirs(fused_dir, exist_ok=True)

    # 1. 针对训练集执行VMD分解
    print("=== 对训练集进行VMD分解 ===")
    step1_vmd_decompose(train_dir, os.path.join(vmd_dir, "train_vmd"))

    # 2. 针对训练集提取IMF token特征
    print("=== 对训练集进行IMF token生成 ===")
    step2_imf_token_representation(os.path.join(vmd_dir, "train_vmd"), imf_dir)

    # 3. 针对训练集融合特征（生成每个文件的融合特征）
    print("=== 对训练集进行特征融合 ===")
    step3_fuse_tokens(CFG, imf_dir, fused_dir)  # 假设CFG作为环境变量或配置对象传入

    # 重复以上步骤，对测试集进行相同处理
    print("=== 对测试集进行VMD分解 ===")
    step1_vmd_decompose(test_dir, os.path.join(vmd_dir, "test_vmd"))
    print("=== 对测试集进行IMF token生成 ===")
    step2_imf_token_representation(os.path.join(vmd_dir, "test_vmd"), imf_dir)
    print("=== 对测试集进行特征融合 ===")
    step3_fuse_tokens(CFG, imf_dir, fused_dir)

    # TEST_CLEAN_DIR = "data/test_clean"
    # SNR_LIST = [20, 10, 0, -5]

    # for snr in SNR_LIST:
    #     noisy_test_dir = f"data/test_SNR_{snr}dB"
    #
    #     make_noisy_testset(
    #         clean_root=TEST_CLEAN_DIR,
    #         noisy_root=noisy_test_dir,
    #         snr_db=snr,
    #         noise_files=None,  # 若使用真实噪声，传入列表
    #         seed=42
    #     )
    #
    #     # step1_vmd_decompose(noisy_test_dir, ...)
    #     # step2_imf_token_representation(...)
    #     # step3_fuse_tokens(...)
    #     # test_model(...)

def load_features(feature_dir, mode="train"):
    """
    从融合特征目录中加载训练或测试特征。
    返回：特征矩阵 X，标签向量 y。
    """
    X_list, Y_list = [], []
    class_dirs = sorted(Path(feature_dir).glob("class_*"))
    for class_dir in class_dirs:
        label = int(class_dir.name.split("_")[-1])
        for npz_file in class_dir.glob(f"*_{mode}.npz"):
            data = np.load(npz_file, allow_pickle=True)
            Z = data["fused_features"]  # 文件级融合特征 (d_model 维向量)
            X_list.append(Z)
            Y_list.append(label)
    X = np.vstack(X_list)
    y = np.array(Y_list)
    print(f"Loaded {len(Y_list)} samples for {mode}. X.shape={X.shape}, y.shape={y.shape}")
    return X, y


def train_model(X_train, y_train, model_name="rf", **model_kwargs):
    clf = build_model(model_name, **model_kwargs)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(report)
    print("\nConfusion Matrix:")
    print(cm)

    return acc, report, cm


def main():
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"
    # 1. 数据准备：划分并复制
    train_dir, test_dir = prepare_data(data_root)

    # 2. 特征提取：VMD分解 + IMF-token + 特征融合
    vmd_output = "outputs/vmd_results"
    imf_output = "outputs/imf_tokens"
    fused_output = "outputs/fused_features"
    extract_and_fuse_features(train_dir, test_dir, vmd_output, imf_output, fused_output)

    # 3. 加载训练集和测试集特征
    X_train, y_train = load_features(fused_output, mode="train")
    X_test, y_test = load_features(fused_output, mode="test")

    # 4. 模型训练：这里以SVM为例
    clf = train_model(X_train, y_train, model_name="rf", n_estimators=500, max_depth=None)

    # 5. 测试评估：在测试集上进行预测
    evaluate_model(clf, X_test, y_test)

if __name__ == "__main__":
    main()
