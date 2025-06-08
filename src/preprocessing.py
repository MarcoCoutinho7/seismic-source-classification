import numpy as np
import pywt
import librosa

def wavelet_denoise(signal, wavelet="db4", level=3):
    coeffs = pywt.wavedec(signal, wavelet, level=level)
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs = [pywt.threshold(c, value=uthresh, mode='soft') for c in coeffs]
    return pywt.waverec(coeffs, wavelet)

def compute_stft(signal, sr, n_fft=512, hop_length=None):
    stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
    stft_db = librosa.amplitude_to_db(np.abs(stft))
    return stft_db
