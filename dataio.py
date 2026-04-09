# dataio.py
import os
import librosa
import numpy as np
from typing import List, Tuple

def load_wav(path: str, sr: int) -> np.ndarray:
    """加载单通道音频为 float32，重采样至 sr。"""
    x, _ = librosa.load(path, sr=sr, mono=True)
    # 这里注意是否有强制重采样的？

    return x

def list_wavs_with_labels(root):
    """
    扫描数据目录并自动分配类别标签。
    要求目录结构形如：
        root/classA/*.wav
        root/classB/*.wav
    返回：
        [(wav_path, label_int), ...]
    """
    items = []
    class_names = sorted([d for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))])
    label_map = {name: idx for idx, name in enumerate(class_names)}
    print("类别映射：", label_map)

    for cname in class_names:
        cdir = os.path.join(root, cname)
        wavs = [os.path.join(cdir, f) for f in os.listdir(cdir) if f.lower().endswith(".wav")]
        for w in wavs:
            items.append((w, label_map[cname]))

    print(f"共发现 {len(items)} 个音频文件。类别分布：")
    counts = {label: 0 for label in label_map.values()}
    for _, y in items:
        counts[y] += 1
    print(counts)
    return items

def add_awgn(signal: np.ndarray, snr_db: float, seed: int = None) -> np.ndarray:
    """
    在 signal 上添加高斯白噪声以达到目标 SNR（dB）。
    - signal: 1D numpy array
    - snr_db: 目标 SNR，单位 dB（正值表示信号强于噪声，例如 +6 dB；负值表示噪声更强）
    - seed: 随机种子（可选）
    返回：噪声化后的信号（float64）
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()

    # 信号功率（均方值）
    p_signal = np.mean(signal.astype(np.float64) ** 2)
    if p_signal <= 0:
        # 信号为空或零向量，直接返回
        return signal.copy()

    # 目标噪声功率
    p_noise = p_signal / (10.0 ** (snr_db / 10.0))

    # 生成高斯噪声（均值0、方差1），然后缩放到目标方差
    noise = rng.standard_normal(size=signal.shape)
    # 当前噪声的方差（近似1），缩放因子：
    current_var = np.mean(noise ** 2)
    # 若 current_var 非常小，避免除以零
    if current_var <= 1e-16:
        current_var = 1.0
    scale = np.sqrt(p_noise / current_var)
    noise_scaled = noise * scale

    noisy_signal = signal.astype(np.float64) + noise_scaled

    # 避免裁剪/溢出：如果信号原本在 -1..1 范围，可能需要做幅度裁剪或归一化
    # 下面不强制裁剪，仅返回浮点数组；若需要可加 clip(-1,1)
    return noisy_signal
