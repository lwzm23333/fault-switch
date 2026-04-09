import librosa
import numpy as np
from pathlib import Path
from vmdpy import VMD
from multiprocessing import Pool, cpu_count
from VMD_fault.Features_list_2 import extract_imf_features, BASIC_IMF_FEATURES
from VMD_fault.features_extractors.base import scan_wavs, save_npz

# 用固定参数的VMD分解，只完成特征提取，不做融合

def vmd_decompose_fixed(
    wav_path,
    K=5,
    alpha=2000,
    tau=0,
    DC=0,
    init=1,
    tol=1e-7,
    sr=16000,
):
    """
    Fixed-parameter VMD (baseline)
    """
    signal, sr0 = librosa.load(wav_path, sr=sr)
    signal = signal - np.mean(signal)

    u, _, _ = VMD(
        signal,
        alpha=alpha,
        tau=tau,
        K=K,
        DC=DC,
        init=init,
        tol=tol
    )

    # u shape: (K, N)
    return u, sr0

def save_imfs_npz(out_dir, wav_path, imfs, sr, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = Path(wav_path).stem
    save_path = out_dir / f"{fname}_imfs.npz"

    # 如果文件已存在，则返回路径但不覆盖
    if save_path.exists():
        print(f"⚙️ 跳过已存在 IMF 文件: {save_path}")
        return save_path

    np.savez(
        save_path,
        imfs=imfs,
        sr=sr,
        label=label
    )
    return save_path

# ------------------ 单文件处理 ------------------
def process_single_file(args):
    wav_path, y, imf_save_dir = args
    fname = Path(wav_path).stem

    # 已存在则跳过
    save_path = Path(imf_save_dir) / f"{fname}_imfs.npz"
    if save_path.exists():
        print(f"⚙️ 跳过已存在 IMF 文件: {save_path}")
        # 直接读取 IMF 并提取特征
        data = np.load(save_path)
        imfs, fs = data['imfs'], data['sr'].item() if hasattr(data['sr'], 'item') else data['sr']
    else:
        imfs, fs = vmd_decompose_fixed(wav_path)
        save_imfs_npz(imf_save_dir, wav_path, imfs, fs, y)

    # 特征提取
    X_imf = extract_imf_features(imfs, fs, BASIC_IMF_FEATURES)
    X = X_imf.flatten()
    print(f"{fname} | label={y} | IMFs={imfs.shape} | feat_dim={X.shape[0]}")
    return X, y

# ------------------ 并行主流程 ------------------
def extract_vmd_fixed_features(data_dir, out_dir, imf_save_dir, num_workers=None):
    wavs = scan_wavs(data_dir)
    args_list = [(wav_path, y, imf_save_dir) for wav_path, y in wavs]

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)  # 保留一个核心给系统

    feats, labels = [], []
    with Pool(num_workers) as pool:
        results = pool.map(process_single_file, args_list)

    for X, y in results:
        feats.append(X)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    return X, y
