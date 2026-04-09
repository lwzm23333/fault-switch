import os
import shutil
import pickle
import numpy as np
from pathlib import Path
from dataset import split_dataset
from config import CFG
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
            dst_file = os.path.join(dst_dir,os.path.basename(filepath))
            if not os.path.exists(dst_file):
                shutil.copy(filepath, dst_file)
    print(f"[prepare_data] Data copied to: {train_dir} and {test_dir}")
    return train_dir, test_dir


def load_features(feature_dir, mode="train"):
    """
    从融合特征目录中加载训练或测试特征。
    返回：特征矩阵 X，标签向量 y。
    """
    X_list, Y_list = [], []
    class_dirs = sorted(Path(feature_dir).glob("class_*"))
    for class_dir in class_dirs:
        label = int(class_dir.name.split("_")[-1])
        for npz_file in class_dir.glob("*.npz"):
            d = np.load(npz_file)
            X_list.append(d["fused_features"])
            Y_list.append(label)
    return np.vstack(X_list), np.array(Y_list)


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
    FUSER_MODEL_PATH = "outputs/models/fuser_weights.pth"

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
    step3_fuse_tokens(CFG, imf_test, fused_test, fuser_state=FUSER_MODEL_PATH)

    # ========= 4. 加载测试特征 =========
    X_test, y_test = load_features(fused_test, mode="test")
    print("X_test mean:", X_test.mean(axis=0)[:10])
    print("X_test std :", X_test.std(axis=0)[:10])

     # ========= 5. 加载训练好的模型 =========
    import pickle
    with open("outputs/models/rf_fuser.pkl", "rb") as f:
        clf = pickle.load(f)
        print(clf)

    # ========= 6. 测试评估 =========
    print(f"\n[TEST MODE] {test_mode.upper()} | SNR = {snr_db}")
    evaluate_model(clf, X_test, y_test)



if __name__ == "__main__":
    # main_test(test_mode="clean")

    # # 高斯白噪声
    main_test(test_mode="noisy", snr_db=3)
    #
    # # ESC-50 城市噪声
    # main_test(test_mode="esc50", snr_db=10)
