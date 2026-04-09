import librosa
import pywt
import numpy as np

def wavelet_decompose_fixed(
    wav_path,
    wavelet="db4",
    level=4,
    sr=None
):
    """
    Fixed wavelet decomposition (baseline)
    """
    signal, sr0 = librosa.load(wav_path, sr=sr)
    signal = signal - np.mean(signal)

    coeffs = pywt.wavedec(signal, wavelet, level=level)

    # coeffs: [cA_L, cD_L, ..., cD1]
    # 统一为 (K, N)
    imfs = []

    for c in coeffs:
        imfs.append(c)

    # 对齐长度（补零到最长）
    max_len = max(len(c) for c in imfs)
    imfs_pad = []

    for c in imfs:
        if len(c) < max_len:
            c = np.pad(c, (0, max_len - len(c)))
        imfs_pad.append(c)

    imfs = np.vstack(imfs_pad)
    return imfs, sr0

from pathlib import Path
from VMD_fault.Features_list_2 import extract_imf_features, BASIC_IMF_FEATURES

def process_single_file_wavelet(args):
    wav_path, y, imf_save_dir, wavelet, level = args
    fname = Path(wav_path).stem
    save_path = Path(imf_save_dir) / f"{fname}_imfs.npz"

    # -------- 1. 读取或分解 --------
    if save_path.exists():
        print(f"⚙️ 跳过已存在 Wavelet IMF 文件: {save_path}")
        data = np.load(save_path)
        imfs = data["imfs"]
        fs = int(data["sr"])
    else:
        imfs, fs = wavelet_decompose_fixed(
            wav_path,
            wavelet=wavelet,
            level=level
        )
        np.savez(
            save_path,
            imfs=imfs,
            sr=fs,
            label=y
        )

    # -------- 2. 特征提取（统一 IMF 特征）--------
    X_imf = extract_imf_features(
        imfs,
        fs,
        BASIC_IMF_FEATURES
    )

    X = X_imf.flatten()

    print(
        f"{fname} | label={y} | "
        f"WaveletBands={imfs.shape} | feat_dim={X.shape[0]}"
    )

    return X, y

from multiprocessing import Pool, cpu_count
from VMD_fault.features_extractors.base import scan_wavs, save_npz

def extract_wavelet_fixed_features_parallel(
    data_dir,
    out_dir,
    imf_save_dir,
    wavelet="db4",
    level=4,
    num_workers=None
):
    wavs = scan_wavs(data_dir)
    args_list = [
        (wav_path, y, imf_save_dir, wavelet, level)
        for wav_path, y in wavs
    ]

    if num_workers is None:
        num_workers = max(cpu_count() - 1, 1)

    print("[INFO] Wavelet-fixed feature extraction started")
    print(f"[INFO] Total samples: {len(wavs)}")
    print(f"[INFO] Wavelet: {wavelet}, level={level}")

    feats, labels = [], []

    with Pool(num_workers) as pool:
        results = pool.map(process_single_file_wavelet, args_list)

    for X, y in results:
        feats.append(X)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    return X, y
