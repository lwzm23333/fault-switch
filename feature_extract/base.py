import os
import numpy as np
from pathlib import Path

def scan_wavs(data_dir):
    """
    扫描 data_dir/class_xxx/*.wav
    """
    wavs = []
    class_dirs = sorted(Path(data_dir).glob("*"))
    for label, cdir in enumerate(class_dirs):
        for wav in cdir.glob("*.wav"):
            wavs.append((str(wav), label))
    return wavs


def save_npz(out_dir, X, y):
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "features.npz")
    np.savez(save_path, X=X, y=y)
    return save_path


def load_npz(out_dir):
    p = os.path.join(out_dir, "features.npz")
    d = np.load(p)
    return d["X"], d["y"]
