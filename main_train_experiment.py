from train_classifier import build_model, crossval_train_eval
import pickle, os
import shutil
import random
import json
from collections import defaultdict
from dataset import split_dataset
#  对比实验

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
            dst_file = os.path.join(dst_dir,os.path.basename(filepath))
            if not os.path.exists(dst_file):
                shutil.copy(filepath, dst_file)
        print(f"[prepare_data] Data copied to: {train_dir} and {test_dir}")
        return train_dir, test_dir


def extract_features_pipeline(method, data_dir, out_dir,imf_save_dir):
    if method == "stft":
        from features_extractors.STFT_feat import extract_stft_features
        return extract_stft_features(data_dir, out_dir)

    elif method == "mfcc":
        from features_extractors.MFCC_feat import extract_mfcc_features
        return extract_mfcc_features(data_dir, out_dir)

    elif method == "vmd_fixed":
        from features_extractors.vmd_feature import extract_vmd_fixed_features
        return extract_vmd_fixed_features(data_dir, out_dir, imf_save_dir)

    elif method == "emd":
        from features_extractors.EMD_feature import emd_features
        return emd_features(data_dir, out_dir)

    elif method == "wavelet":
        from features_extractors.wavelet_feature import wavelet_features
        return wavelet_features(data_dir, out_dir)

    elif method == "imf_token":
        from features_extractors.rmsa_imf_fuse_feature import extract_imf_token_features
        return extract_imf_token_features(data_dir, out_dir)

    else:
        raise ValueError(f"Unknown method: {method}")


def train_one_method(method, train_dir, model_name):
    print(f"\n===== Training {method} + {model_name} =====")

    feat_dir = f"outputs/different_features/{method}/train"
    imf_save_dir = f"outputs/vmd_results/{method}/train"
    os.makedirs(feat_dir, exist_ok=True)

    X_train, y_train = extract_features_pipeline(
        method, train_dir, feat_dir,imf_save_dir
    )

    # CV 评估（论文可用）
    cv_results = crossval_train_eval(
        X_train, y_train,
        model_name=model_name,
        n_splits=5
    )
    print("[CV results]", cv_results)

    clf = build_model(model_name)
    clf.fit(X_train, y_train)

    model_path = f"outputs/models/{method}_{model_name}.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"Model saved to {model_path}")

def main_train_all(model):
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"
    train_dir, test_dir = prepare_data(data_root)

    # methods = ["stft", "mfcc", "vmd_fixed", "imf_token","wavelet"]
    methods = ["vmd_fixed"]

    for m in methods:
        train_one_method(m, train_dir, model_name=model)

if __name__ == "__main__":
    main_train_all("rf")
