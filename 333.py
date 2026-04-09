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
from noise_data import make_noisy_testset, collect_esc50_noise

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


def load_features(feature_dir, mode="train"):
    """
    从融合特征目录中加载训练或测试特征。
    返回：特征矩阵 X，标签向量 y。
    """
    X_list, Y_list = [], []
    class_dirs = sorted(Path(feature_dir).glob("class_*"))
    for class_dir in class_dirs:
        for npz_file in class_dir.glob("*.npz"):
            data = np.load(npz_file, allow_pickle=True)
            X_list.append(data["fused_features"])
            Y_list.append(int(data["labels"]))

    X = np.vstack(X_list)
    y = np.array(Y_list)
    print(f"Loaded {len(Y_list)} samples for {mode}. X.shape={X.shape}, y.shape={y.shape}")
    return X, y


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


def main_test(test_mode="clean", snr_db=None):
    """
    test_mode:
        - "clean"  : 干净测试集
        - "noisy"  : 高斯白噪声测试集
        - "esc50"  : ESC-50 城市环境噪声测试集
    """
    assert test_mode in ["clean", "noisy", "esc50"]

    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"

    # ========= 1. 构建测试集（只使用既定划分） =========
    _, test_dir = prepare_data(data_root)

    # ========= 2. 根据测试模式生成测试数据 =========
    if test_mode == "noisy":
        assert snr_db is not None, "snr_db must be specified for noisy test mode"

        noisy_test_dir = f"data/test_AWGN_{snr_db}dB"

        make_noisy_testset(
            clean_root=test_dir,
            noisy_root=noisy_test_dir,
            snr_db=snr_db,
            noise_pool=None,     # === AWGN ===
            seed=42
        )

        test_dir = noisy_test_dir
        tag = f"AWGN_{snr_db}dB"

    elif test_mode == "esc50":
        assert snr_db is not None, "snr_db must be specified for esc50 test mode"

        esc50_root = "D:/Work/pythonproject/Datasets/ESC-50"
        target_sr = CFG.sample_rate

        # 加载并重采样 ESC-50 噪声
        noise_pool = collect_esc50_noise(
            esc50_root=esc50_root,
            target_sr=target_sr
        )

        noisy_test_dir = f"data/test_ESC50_{snr_db}dB"

        make_noisy_testset(
            clean_root=test_dir,
            noisy_root=noisy_test_dir,
            snr_db=snr_db,
            noise_pool=noise_pool,   # === 真实城市噪声 ===
            seed=42
        )

        test_dir = noisy_test_dir
        tag = f"ESC50_{snr_db}dB"

    else:
        # clean test
        tag = "clean"

    # ========= 3. 测试集特征提取（严格前向） =========
    vmd_test = f"outputs/vmd_results/test_{tag}"
    imf_test = f"outputs/imf_tokens/test_{tag}"
    fused_test = f"outputs/fused_features/test_{tag}"

    step1_vmd_decompose(test_dir, vmd_test)
    step2_imf_token_representation(vmd_test, imf_test)
    step3_fuse_tokens(CFG, imf_test, fused_test, fuser_state="outputs/models/fuser_weights.pth")

    # ========= 4. 加载测试特征 =========
    X_test, y_test = load_features(fused_test, mode="test")
    FUSER_MODEL_PATH = "outputs/models/fuser_weights.pth"

    # 如果文件不存在，给个提示
    if not os.path.exists(FUSER_MODEL_PATH):
        print(f"❌ 错误: 找不到 Fuser 权重文件 {FUSER_MODEL_PATH}")
        return


     # ========= 5. 加载训练好的模型 =========
    import pickle
    with open("outputs/models/rf_fuser.pkl", "rb") as f:
        clf = pickle.load(f)

        # ========= [DEBUG 关键检查区域] =========
        print("\n" + "=" * 30)
        print("数据一致性检查 (Debugging)")
        print("=" * 30)

        # 1. 检查真实标签是否正确加载 (是否包含 0-9)
        unique_y, counts_y = np.unique(y_test, return_counts=True)
        print(f"y_test 真实标签分布: {dict(zip(unique_y, counts_y))}")
        print(f"y_test 前 20 个值: {y_test[:20]}")

        # 2. 检查模型预测值的分布 (看看是不是真的全预测为 9)
        y_pred = clf.predict(X_test)
        unique_p, counts_p = np.unique(y_pred, return_counts=True)
        print(f"y_pred 模型预测分布: {dict(zip(unique_p, counts_p))}")
        print(f"y_pred 前 20 个值: {y_pred[:20]}")

        # 3. 检查特征数值是否异常
        print(f"特征矩阵是否有 NaN: {np.isnan(X_test).any()}")
        print(f"特征矩阵均值: {X_test.mean():.4f}, 标准差: {X_test.std():.4f}")
        print(f"前 2 个样本的前 5 维特征:\n{X_test[:2, :5]}")
        print("=" * 30 + "\n")

        # ========= 6. 测试评估 =========
        print(f"\n[TEST MODE] {test_mode.upper()} | SNR = {snr_db}")
        evaluate_model(clf, X_test, y_test)



if __name__ == "__main__":
    # main_test(test_mode="noisy", snr_db=10)
    # main_test(test_mode="noisy", snr_db=-6)
    # main_test(test_mode="noisy", snr_db=0)
    main_test(test_mode="noisy", snr_db=6)

