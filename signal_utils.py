# utils/signal_utils.py
import numpy as np

def mad_sigma(x: np.ndarray) -> float:
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def spectral_entropy(x, fs, nfft=2048, eps=1e-12):
    X = np.abs(np.fft.rfft(x, n=nfft))**2
    P = X / (np.sum(X)+eps)
    H = -np.sum(P * np.log(P+eps)) / np.log(len(P))
    return H

def snr_estimate(x, noise_floor_ratio=0.1):
    power = np.mean(x**2) + 1e-12
    noise = np.quantile((x**2), noise_floor_ratio) + 1e-12
    return 10*np.log10(power/noise)

def band_energy(x, fs, fband):
    N = len(x)
    freqs = np.fft.rfftfreq(N, 1/fs)
    X = np.fft.rfft(x)
    m = (freqs >= fband[0]) & (freqs <= fband[1])
    return np.sum(np.abs(X[m])**2)
