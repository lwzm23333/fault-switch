import numpy as np
import os
import shutil
from preprocessing import preprocess
from config import CFG
from dataset import split_dataset
# from vmd.adaptive_vmd import adaptive_vmd
# from vmd.imf_select import select_imf, weighted_reconstruct
# from vmd.imf_features import imf_features
# from features.fuzzy_selector import select_features_fuzzy
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import joblib
from adaptive_vmd import step1_vmd_decompose
from extract_features import step2_extract_features_plain
from features_select import step3_feature_selection_fuzzy
from extract_features_token import step2_imf_token_representation
from features_fuser import step3_fuse_tokens

# --------------------------------------------------
# 数据集构造
# --------------------------------------------------
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

# --------------------------------------------------
# 特征数据集构建
# --------------------------------------------------

def feature_select(
        vmd_save_dir: str,
        feat_excel_path: str,
        feat_npz_path: str,
        selected_npz_path: str,
        all_npz_path:str,
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


# --------------------------------------------------
# 主训练函数
# --------------------------------------------------

def main_train(wav_list, labels, fs):
    X = []
    for wav in wav_list:
        feat = process_file_train(wav, fs)
        X.append(feat)

    X = np.array(X)
    y = np.array(labels)

    X_fs, idx = select_features_fuzzy(X, top_m=40)

    scaler = StandardScaler().fit(X_fs)
    Xs = scaler.transform(X_fs)

    clf = SVC(kernel='rbf', C=1, gamma='scale', probability=True)
    clf.fit(Xs, y)

    joblib.dump((scaler, clf, idx), "model.pkl")


# --------------------------------------------------
# Demo 主程序（构造示例数据）
# --------------------------------------------------
if __name__ == "__main__":
    fs = CFG.seg.fs_target  # 目标采样率
    data_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data"
    train_root = "D:/Work/pythonproject/Datasets/Data_Sets/Switch_machine_audio/data/train"
    vmd_save_dir = "outputs/train"
    # ------筛选部分-----
    feat_all_excel_path = "outputs/feat_all_excel.xlsx"
    feat_npz_path = "outputs/train_feat"
    all_npz_path = "outputs/train_feat/all_file_features.npz"
    selected_npz_path = "outputs/train_feat/selected_features.npz"
    top_k = 30         # 自定义
    # ----- 正式工程 ------
    feat_excel_path = "outputs/feat_excel.xlsx"
    IMF_npz_path = "outputs/train_IMF_feat"
    fuser_npz_path = "outputs/fuser_npz_path"

    # # step0 数据集的划分
    # wav_files, class_names = load_dataset(data_root)
    # train_list, test_list = split_dataset(wav_files, test_ratio=0.3, seed=42, save_path="data/split.json")
    #
    # # 将训练集和测试集文件复制到指定文件夹
    # train_root = os.path.join(data_root, "Train")
    # test_root = os.path.join(data_root, "Test")
    #
    # for dataset, root in [(train_list, train_root), (test_list, test_root)]:
    #     for filepath, label in dataset:
    #         class_name = class_names[label]
    #         dst_dir = os.path.join(root, class_name)
    #         os.makedirs(dst_dir, exist_ok=True)
    #         dst_file = os.path.join(dst_dir, os.path.basename(filepath))
    #         shutil.copy(filepath, dst_file)
    #
    # print(f"训练集已复制到: {train_root}")
    # print(f"测试集已复制到: {test_root}")
    #
    # # step1 数据集构造
    # feats = process_file_train(train_root=train_root,
    #                            vmd_save_dir=vmd_save_dir,
    #                            feat_excel_path=feat_excel_path,
    #                             feat_npz_path=feat_npz_path,
    #                            selected_npz_path=selected_npz_path,
    #                            top_k=top_k)
    #
    # # step2 训练
    #
    # # step1 训练集自适应VMD分解            adaptive_vmd.py
    # step1_vmd_decompose(train_root, vmd_save_dir)
    #
    # # 对所有特征值提取后进行筛选，仅用于理解特征贡献度与构建 IMF-Token 结构，不参与最终训练。
    # # (即包含step2_extract_features_plain和step3_feature_selection_fuzzy)
    # feature_select(vmd_save_dir, feat_all_excel_path, feat_npz_path, all_npz_path, selected_npz_path, top_k)
    #
    #
    # # step2 对IMF组的特征提取             extract_features.py
    # step2_extract_features_plain(vmd_save_dir, feat_all_excel_path, feat_npz_path)
    #
    # # step3 读取excel特征，进行特征筛选     features_select.py
    # step3_feature_selection_fuzzy(all_npz_path, selected_npz_path, top_k)
    #
    # # step2 特征提取及IMF-token
    # step2_imf_token_representation(vmd_save_dir, IMF_npz_path)

    # step3 特征融合                     features_fuser.py
    step3_fuse_tokens(CFG, IMF_npz_path, fuser_npz_path)

    # step4 模型训练
