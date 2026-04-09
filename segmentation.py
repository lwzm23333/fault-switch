# segmentation.py

# 功能：对一维时域信号进行分段（frame）和基于 OLA (Overlap-Add) 的重构

import numpy as np
from scipy.signal.windows import hann

def frame_signal(x, fs, seg_len_s=1.0, hop_ratio=0.5, window="hann"):
    """
        将长信号切分为重叠的分帧信号，并加上窗函数。

        参数
        ----------
        x : np.ndarray
            输入一维信号（音频或振动信号）。
        fs : int
            采样率 (Hz)。
        seg_len_s : float, 默认 1.0
            每一帧的长度（秒）。
        hop_ratio : float, 默认 0.5
            帧移占帧长的比例，0.5 表示 50% 重叠。
        window : str, 默认 "hann"
            窗函数类型，目前支持 "hann"，否则为矩形窗。

        返回
        ----------
        frames : np.ndarray, shape = [num_frames, nper]
            分帧后的信号矩阵，每行一帧。
        idxs : np.ndarray
            每帧起始位置（相对于原信号的索引）。
        win : np.ndarray
            窗函数本身（长度 nper）。
        hop : int
            帧移点数。
        """
    # 每帧点数 = 秒数 × 采样率
    nper = int(seg_len_s * fs)
    # 帧移点数 = 帧长 × hop_ratio
    hop = int(nper * hop_ratio)
    # 选择窗函数：默认 Hann 窗，否则矩形窗（全 1）
    if window == "hann":
        win = np.sqrt(hann(nper, sym=False))
    else:
        win = np.ones(nper)
    frames = []    # 存储分帧后的结果
    idxs = []      # 存储每一帧的起始点
    # 以 hop 为步长滑动窗口
    for start in range(0, max(1, len(x) - nper + 1), hop):
        seg = x[start:start+nper]
        if len(seg) < nper:      # 最后一帧可能不足，补零
            pad = np.zeros(nper - len(seg))
            seg = np.concatenate([seg, pad])
        # 乘以窗函数，抑制边缘效应
        frames.append(seg * win)
        idxs.append(start)
    return np.array(frames), np.array(idxs), win, hop

def ola_reconstruct(frames, win, hop, frames_are_windowed=True):
    frames = np.asarray(frames)
    win = np.asarray(win)
    n_frames, nper = frames.shape
    total_len = hop * (n_frames - 1) + nper
    y = np.zeros(total_len)
    wsum = np.zeros(total_len)

    if frames_are_windowed:
        for i in range(n_frames):
            start = i * hop
            y[start:start+nper] += frames[i]
            wsum[start:start+nper] += win
    else:
        for i in range(n_frames):
            start = i * hop
            y[start:start+nper] += frames[i] * win
            wsum[start:start+nper] += win**2

    # ✅ 防止 wsum 太小导致开头/结尾爆炸
    eps = 1e-6
    wsum[wsum < eps] = np.min(wsum[wsum >= eps])  # 用最小非零值填充
    y = y / wsum

    # ✅ 可选归一化（防止能量漂移）
    y = y / (np.max(np.abs(y)) + 1e-12)
    return y


def ola_selfcheck(fs=16000, N=16000, seg_len_s=1.0, hop_ratio=0.5, window="hann"):
    """
    检查 OLA 重构是否能量守恒
    """
    import numpy as np
    t = np.arange(N) / fs
    sig = np.zeros(N)
    sig[N//2] = 1.0   # 单位脉冲
    sig += 0.5 * np.sin(2*np.pi*200*t)  # 加个正弦

    frames, _, win, hop = frame_signal(sig, fs, seg_len_s, hop_ratio, window)
    rec = ola_reconstruct(frames, win, hop)

    ER = np.sum(rec ** 2) / (np.sum(sig ** 2) + 1e-12)
    max_err = np.max(np.abs(sig - rec))
    print(f"[OLA自检] ER={ER:.4f}, max_err={max_err:.4e}")
    return sig, rec