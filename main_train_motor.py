from train_classifier import build_model, crossval_train_eval
import pickle
import os
import shutil
import random
import json
from collections import defaultdict
import librosa
import soundfile as sf
import numpy as np

# ===================== 核心配置（根据你的需求调整） =====================
SLICE_DURATION = 5  # 5s/切片
SLICED_DATA_DIR = "D:/Work/pythonproject/Datasets/Data_Sets/MPSS/Data"  # 切片保存根目录
DATASET_ROOT = "D:/Work/pythonproject/Datasets/Data_Sets/MPSS/PCB microphone and phone"  # 数据集根目录
# 长音频划分配置（每类6条长音频：5条训练，1条测试）
TRAIN_LONG_AUDIO_NUM = 5  # 每类用于训练的长音频数量
TEST_LONG_AUDIO_NUM = 1  # 每类用于测试的长音频数量
SEED = 42  # 固定随机种子，保证划分结果可复现
SAMPLE_RATE = 44100  # 固定采样率为44100Hz


# ===================== 工具函数 =====================
def extract_label_from_filename(filename):
    """提取故障类型标签（b1/b2/b3/n），忽略f/l/r"""
    basename = os.path.splitext(filename)[0].lower()
    if "b1" in basename:
        return "b1"
    elif "b2" in basename:
        return "b2"
    elif "b3" in basename:
        return "b3"
    elif "n" in basename:
        return "n"
    else:
        raise ValueError(f"无法从 {filename} 提取标签（b1/b2/b3/n）")


def slice_audio_file(audio_path, output_dir, label, slice_duration=5):
    """固定采样率44100Hz，切片并保存"""
    # 加载音频并强制转为44100Hz采样率
    y, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    total_duration = len(y) / SAMPLE_RATE
    print(f"[切片] {os.path.basename(audio_path)} | 采样率: {SAMPLE_RATE} Hz | 时长: {total_duration:.1f} s")

    # 计算切片的样本点数（5s * 44100 = 220500个样本点）
    slice_samples = int(slice_duration * SAMPLE_RATE)
    total_slices = len(y) // slice_samples

    if total_slices == 0:
        print(f"警告：{audio_path} 时长过短（{total_duration:.1f}s），无法生成 {slice_duration}s 切片，跳过")
        return []

    sliced_files = []
    # 创建标签对应的保存目录
    label_dir = os.path.join(output_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    # 生成切片文件
    audio_basename = os.path.splitext(os.path.basename(audio_path))[0]
    for i in range(total_slices):
        start = i * slice_samples
        end = start + slice_samples
        slice_y = y[start:end]

        # 切片文件名（包含切片序号）
        slice_filename = f"{audio_basename}_slice_{i:04d}.wav"
        slice_path = os.path.join(label_dir, slice_filename)

        # 保存切片（固定44100Hz采样率）
        sf.write(slice_path, slice_y, SAMPLE_RATE)
        sliced_files.append(slice_path)

    print(f"[切片完成] 生成 {total_slices} 个 {slice_duration}s 切片")
    return sliced_files


# ===================== 先划分长音频，再切片 =====================
def split_long_audio_and_slice(root, force=False):
    """
    核心逻辑：
    1. 按类别收集所有长音频；
    2. 每类划分5条训练、1条测试长音频；
    3. 分别切片，保存到SLICED_DATA_DIR下的train/test目录
    """
    # 最终的训练/测试切片目录
    train_sliced_dir = os.path.join(SLICED_DATA_DIR, "train")
    test_sliced_dir = os.path.join(SLICED_DATA_DIR, "test")

    # 若已存在切片数据且不强制重新处理，直接返回
    if (
        os.path.exists(train_sliced_dir)
        and os.path.exists(test_sliced_dir)
        and not force
    ):
        train_has_wav = False
        test_has_wav = False

        for _, _, files in os.walk(train_sliced_dir):
            if any(f.lower().endswith(".wav") for f in files):
                train_has_wav = True
                break

        for _, _, files in os.walk(test_sliced_dir):
            if any(f.lower().endswith(".wav") for f in files):
                test_has_wav = True
                break

        if train_has_wav and test_has_wav:
            print(f"[split_long_audio_and_slice] 使用已存在的切片数据: {SLICED_DATA_DIR}")
            return train_sliced_dir, test_sliced_dir
        else:
            print("[split_long_audio_and_slice] 检测到切片目录为空或不完整，重新生成切片")

    # ================== 清空并重新生成切片 ==================
    if os.path.exists(SLICED_DATA_DIR):
        shutil.rmtree(SLICED_DATA_DIR)
    os.makedirs(train_sliced_dir, exist_ok=True)
    os.makedirs(test_sliced_dir, exist_ok=True)

    # 步骤1：按类别收集所有长音频路径
    class_long_audio = defaultdict(list)  # {标签: [长音频路径1, 长音频路径2,...]}
    for root_dir, _, files in os.walk(root):
        for file in files:
            if file.lower().endswith(".wav"):
                audio_path = os.path.join(root_dir, file)
                try:
                    label = extract_label_from_filename(file)
                    class_long_audio[label].append(audio_path)
                except ValueError as e:
                    print(f"跳过无效文件 {audio_path}: {e}")
                    continue

    # 步骤2：检查每类长音频数量（确保至少有6条）
    for label, audio_list in class_long_audio.items():
        if len(audio_list) < TRAIN_LONG_AUDIO_NUM + TEST_LONG_AUDIO_NUM:
            raise ValueError(
                f"类别 {label} 仅找到 {len(audio_list)} 条长音频，无法划分 {TRAIN_LONG_AUDIO_NUM} 训练 + {TEST_LONG_AUDIO_NUM} 测试"
            )
        if len(audio_list) != 6:
            print(f"警告：类别 {label} 找到 {len(audio_list)} 条长音频（预期6条），请核对数据集！")

    for label, audio_list in class_long_audio.items():
        print(f"[DEBUG] 类别 {label} 长音频列表：")
        for p in audio_list:
            print("   ", os.path.basename(p))
    # 步骤3：划分训练/测试长音频并切片
    random.seed(SEED)  # 固定种子，结果可复现
    for label, audio_list in class_long_audio.items():
        # 随机打乱长音频列表（保证划分随机性）
        random.shuffle(audio_list)
        # 划分训练/测试长音频
        train_long_audio = audio_list[:TRAIN_LONG_AUDIO_NUM]
        test_long_audio = audio_list[TRAIN_LONG_AUDIO_NUM:TRAIN_LONG_AUDIO_NUM + TEST_LONG_AUDIO_NUM]

        print(f"\n[类别 {label}] 训练长音频: {len(train_long_audio)} 条 | 测试长音频: {len(test_long_audio)} 条")

        # 训练长音频切片（保存到train目录）
        for audio_path in train_long_audio:
            slice_audio_file(audio_path, train_sliced_dir, label, SLICE_DURATION)
        # 测试长音频切片（保存到test目录）
        for audio_path in test_long_audio:
            slice_audio_file(audio_path, test_sliced_dir, label, SLICE_DURATION)

    print(f"\n[完成] 训练切片目录: {train_sliced_dir} | 测试切片目录: {test_sliced_dir}")
    return train_sliced_dir, test_sliced_dir


# ===================== 加载切片后的训练/测试集 =====================
def load_sliced_dataset(sliced_dir):
    """加载切片后的数据集（返回 (文件路径, 标签索引) 列表 + 类别名）"""
    class_names = sorted([d for d in os.listdir(sliced_dir) if os.path.isdir(os.path.join(sliced_dir, d))])
    label_to_idx = {label: idx for idx, label in enumerate(class_names)}

    dataset = []
    for label in class_names:
        label_dir = os.path.join(sliced_dir, label)
        for file in os.listdir(label_dir):
            if file.lower().endswith(".wav"):
                file_path = os.path.join(label_dir, file)
                dataset.append((file_path, label_to_idx[label]))

    print(f"[加载切片数据] {sliced_dir} | 总样本数: {len(dataset)} | 类别: {class_names}")
    return dataset, class_names


# ===================== 数据准备主函数 =====================
def prepare_data(data_root=DATASET_ROOT, force=False):
    """数据准备：先划分长音频，再切片，返回训练/测试目录"""
    train_sliced_dir, test_sliced_dir = split_long_audio_and_slice(data_root, force=force)
    return train_sliced_dir, test_sliced_dir


# ===================== 特征提取 & 训练逻辑 =====================
def extract_features_pipeline(method, data_dir, out_dir, imf_save_dir):
    """特征提取管道（复用原有逻辑）"""
    os.makedirs(out_dir, exist_ok=True)
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
        raise ValueError(f"未知特征提取方法: {method}")


def train_one_method(method, train_dir, model_name):
    """单方法训练逻辑"""
    print(f"\n===== 训练 {method} + {model_name} =====")

    # 特征提取目录
    feat_dir = f"outputs/motor_features/{method}/train"
    imf_save_dir = f"outputs/motor_IMF/{method}/train"
    os.makedirs(feat_dir, exist_ok=True)

    # 提取特征
    X_train, y_train = extract_features_pipeline(method, train_dir, feat_dir, imf_save_dir)

    # 交叉验证评估
    cv_results = crossval_train_eval(
        X_train, y_train,
        model_name=model_name,
        n_splits=5
    )
    print("[交叉验证结果]", cv_results)

    # 训练并保存模型
    clf = build_model(model_name)
    clf.fit(X_train, y_train)

    model_path = f"outputs/motor_models/{method}_{model_name}.pkl"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)

    print(f"模型已保存至: {model_path}")


def main_train_all(model):
    """主训练函数"""
    # 数据准备（划分长音频+切片）
    train_dir, test_dir = prepare_data(data_root=DATASET_ROOT)

    # 待训练的特征方法列表
    methods = ["vmd_fixed"]  # 可扩展为 ["stft", "mfcc", "vmd_fixed", "imf_token","wavelet"]

    # 逐个训练
    for m in methods:
        train_one_method(m, train_dir, model_name=model)


if __name__ == "__main__":
    # 训练随机森林（rf）模型，可改为svm、knn等
    main_train_all("rf")