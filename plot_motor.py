import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# 配置参数
# ----------------------------
data_dir = Path("D:/Work/pythonproject/Datasets/Data_Sets/MPSS/PCB microphone and phone")  # 替换为你的 Dataset 1 音频路径
segment_duration = 5.0  # 秒，绘制的短时片段长度
sr_target = 44100       # 假设采样率，如果你的数据不同请调整
labels_map = {
    "n": "Normal",
    "b1": "Magnet fracture",
    "b2": "Excess Hall adhesive",
    "b3": "Bearing too tight"
}

# 配色（好看，论文常用）
colors = {
    "n": "#2e92ce",   # 蓝
    "b1": "#2e92ce",  # 紫
    "b2": "#2e92ce",  # 绿
    "b3": "#2e92ce",  # 红
}

# ----------------------------
# 功能函数
# ----------------------------
def load_segment(wav_path, segment_duration=5.0, sr_target=44100):
    """读取音频，截取前 segment_duration 秒"""
    signal, sr = sf.read(wav_path, dtype='float32')
    if signal.ndim > 1:
        signal = np.mean(signal, axis=1)  # 转为单通道
    # 重采样（如果需要）
    if sr != sr_target:
        import librosa
        signal = librosa.resample(signal, orig_sr=sr, target_sr=sr_target)
        sr = sr_target
    n_samples = int(segment_duration * sr)
    return signal[:n_samples], sr

# ----------------------------
# 读取四类样本
# ----------------------------
sample_signals = {}
for label in labels_map.keys():
    wav_files = list(data_dir.glob(f"{label}*.wav"))
    if not wav_files:
        raise ValueError(f"No wav file found for label {label}")
    # 选第一条音频
    sample_signals[label], sr = load_segment(wav_files[0], segment_duration, sr_target)

# ----------------------------
# 绘制时域图
# ----------------------------
plt.figure(figsize=(12, 6))
for i, (label, signal) in enumerate(sample_signals.items(), 1):
    plt.subplot(2, 2, i)
    time_axis = np.arange(len(signal)) / sr
    plt.plot(time_axis, signal, color=colors[label], linewidth=1)
    plt.title(labels_map[label], fontsize=12)
    plt.xlabel("Time [s]", fontsize=10)
    plt.ylabel("Amplitude", fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()

plt.suptitle("Representative Short-Time Waveforms of Motor States", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
