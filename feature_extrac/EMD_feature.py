import librosa
import numpy as np
from PyEMD import EMD

def emd_decompose_fixed(
    wav_path,
    max_imf=5,
    sr=None
):
    """
    Fixed EMD decomposition (baseline)
    """
    signal, sr0 = librosa.load(wav_path, sr=sr)
    signal = signal - np.mean(signal)

    emd = EMD()
    imfs = emd.emd(signal)

    # 统一 IMF 数量
    if imfs.shape[0] >= max_imf:
        imfs = imfs[:max_imf]
    else:
        # 不足补零，保证维度一致
        pad = np.zeros((max_imf - imfs.shape[0], imfs.shape[1]))
        imfs = np.vstack([imfs, pad])

    return imfs, sr0

from pathlib import Path

def save_imfs_npz(out_dir, wav_path, imfs, sr, label):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fname = Path(wav_path).stem
    save_path = out_dir / f"{fname}_imfs.npz"

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

from VMD_fault.Features_list_2 import extract_imf_features, BASIC_IMF_FEATURES

def process_single_file_emd(args):
    wav_path, y, imf_save_dir, max_imf = args
    fname = Path(wav_path).stem
    save_path = Path(imf_save_dir) / f"{fname}_imfs.npz"

    # -------- 1. 读取或分解 --------
    if save_path.exists():
        print(f"⚙️ 跳过已存在 IMF 文件: {save_path}")
        data = np.load(save_path)
        imfs = data["imfs"]
        fs = int(data["sr"])
    else:
        imfs, fs = emd_decompose_fixed(
            wav_path,
            max_imf=max_imf
        )
        save_imfs_npz(imf_save_dir, wav_path, imfs, fs, y)

    # -------- 2. 特征提取（与 VMD 完全一致）--------
    X_imf = extract_imf_features(
        imfs,
        fs,
        BASIC_IMF_FEATURES
    )

    X = X_imf.flatten()

    print(
        f"{fname} | label={y} | "
        f"IMFs={imfs.shape} | feat_dim={X.shape[0]}"
    )

    return X, y

from multiprocessing import Pool, cpu_count
from VMD_fault.features_extractors.base import scan_wavs, save_npz

def extract_emd_fixed_features_parallel(
    data_dir,
    out_dir,
    imf_save_dir,
    max_imf=5,
    num_workers=None
):
    wavs = scan_wavs(data_dir)
    args_list = [
        (wav_path, y, imf_save_dir, max_imf)
        for wav_path, y in wavs
    ]

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    print("[INFO] EMD-fixed feature extraction started")
    print(f"[INFO] Total samples: {len(wavs)}")
    print(f"[INFO] Using {num_workers} workers")

    feats, labels = [], []

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file_emd, args_list)

    for X, y in results:
        feats.append(X)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    return X, y
