import librosa
import numpy as np
from .base import scan_wavs, save_npz
from scipy.signal import stft
def extract_stft_features(data_dir, out_dir, sr=44100):
    wavs = scan_wavs(data_dir)
    feats, labels = [], []

    for wav_path, y in wavs:
        sig, _ = librosa.load(wav_path, sr=sr)
        S = np.abs(librosa.stft(sig, n_fft=512))
        feat = np.nan_to_num(np.mean(S))
        feats.append(feat)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    print("STFT features:", X.shape)
    return X, y

# def extract_stft_features(data_dir, out_dir, sr=44100):
    wavs = scan_wavs(data_dir)
    feats, labels = [], []

    for wav_path, y in wavs:
        sig, _ = librosa.load(wav_path, sr=sr)

        f, _, Zxx = stft(
            sig,
            fs=sr,
            nperseg=1024,
            noverlap=512,
            window="hann"
        )

        mag = np.abs(Zxx)

        # ---------- STFT frame-level features ----------
        energy = np.sum(mag ** 2, axis=0)

        centroid = np.sum(f[:, None] * mag, axis=0) / (np.sum(mag, axis=0) + 1e-6)

        low = mag[f <= 100].sum(axis=0)
        high = mag[(f > 100) & (f <= 200)].sum(axis=0)
        ratio = low / (high + 1e-6)

        frame_feat = np.vstack([energy, centroid, ratio])  # (3, T)

        # ---------- temporal pooling ----------
        feat = np.concatenate([
            frame_feat.mean(axis=1),
            frame_feat.std(axis=1),
            frame_feat.max(axis=1)
        ])  # 9-D

        feats.append(feat)
        labels.append(y)

    X = np.vstack(feats)
    y = np.array(labels)

    save_npz(out_dir, X, y)
    print("STFT features:", X.shape)
    return X, y

import librosa
import numpy as np
from .base import scan_wavs, save_npz


def extract_stft_features(data_dir, out_dir, sr=44100, n_fft=1024, hop_length=512):
    """
    STFT baseline feature extractor (for RF baseline)

    Each wav -> fixed-length feature vector
    Features: band-wise STFT magnitude statistics
    """
    wavs = scan_wavs(data_dir)
    feats, labels = [], []

    # 频带划分（可在论文中解释）
    freq_bands = [
        (0, 100),
        (100, 300),
        (300, 800),
        (800, 2000)
    ]

    for wav_path, y in wavs:
        sig, _ = librosa.load(wav_path, sr=sr)

        # STFT
        S = np.abs(librosa.stft(
            sig,
            n_fft=n_fft,
            hop_length=hop_length,
            window='hann'
        ))

        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

        feat = []
        for f_low, f_high in freq_bands:
            idx = (freqs >= f_low) & (freqs < f_high)
            band_energy = S[idx, :]

            if band_energy.size == 0:
                feat.extend([0.0, 0.0])
            else:
                feat.append(np.mean(band_energy))

        feats.append(feat)
        labels.append(y)

    X = np.asarray(feats, dtype=np.float32)
    y = np.asarray(labels)

    save_npz(out_dir, X, y)
    print("STFT baseline features:", X.shape)

    return X, y
