# preprocessing.py 音频预处理
import numpy as np
from scipy.signal import detrend

def dc_remove(x: np.ndarray) -> np.ndarray:
    """
    去直流：将信号的均值移除，使其在零附近对称分布。
    x：数据
    返回
    ----------
    np.ndarray
        与 x 形状相同的数组；已移除均值（DC 分量）。
    """
    return detrend(x, type='constant')

def amp_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    幅值归一化：将信号按“最大绝对值”缩放到 [-1, 1] 区间（或更接近）。
    x :param eps:
    :return:
    """
    m = np.max(np.abs(x)) + eps
    return x / m

def preprocess(x: np.ndarray) -> np.ndarray:
    """
        预处理流水线：去直流 + 幅值归一化。

        参数
        ----------
        x : np.ndarray
            原始时间序列（通常为 1D 波形）。建议为实数类型。

        返回
        ----------
        np.ndarray
            预处理后的信号：先去直流，再按最大绝对值归一化。
    """
    # Step 1: 去直流，使信号均值为 0（或极接近 0）
    x = dc_remove(x)
    # Step 2: 按最大绝对值归一化，避免量纲差异影响下游处理
    x = amp_norm(x)
    return x

