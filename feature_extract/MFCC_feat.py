import librosa
import numpy as np
from .base import scan_wavs, save_npz

def extract_mfcc_features(
    data_dir,
    out_dir,
    sr=44100,
    n_mfcc=12
):
    wavs = scan_wavs(data_dir)
    feats, labels = [], []

    for wav_path, y in wavs:
        sig, _ = librosa.load(wav_path, sr=sr)

        mfcc = librosa.feature.mfcc(
            y=sig,
            sr=sr,
            n_mfcc=n_mfcc)  # (n_mfcc, T)

        feat = np.concatenate([
            mfcc.mean(axis=1)])

        feats.append(feat)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    print("MFCC features:", X.shape)
    return X, y
