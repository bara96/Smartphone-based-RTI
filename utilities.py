# Import required modules
import os
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from scipy import signal


# plot signal waves
def plot_waves(sub_wave_matrix, wave_matrix, x_corr):
    plt.plot(sub_wave_matrix)
    plt.title("Sub-Wave Signal")
    plt.show()

    plt.plot(wave_matrix)
    plt.title("Wave Signal")
    plt.show()

    # normalize the cross correlation
    plt.plot(x_corr)
    plt.title("Cross-Correlation Plot")
    plt.show()


# return audio cross correlation
def find_audio_correlation(sub_wave_matrix, wave_matrix, plot=False):
    sub_wave_matrix = sub_wave_matrix.flatten()
    wave_matrix = wave_matrix.flatten()
    x_corr = signal.correlate(sub_wave_matrix - np.mean(sub_wave_matrix), wave_matrix - np.mean(wave_matrix),
                              mode='valid') / (np.std(sub_wave_matrix) * np.std(wave_matrix) * len(wave_matrix))
    if plot:
        plot_waves(sub_wave_matrix, wave_matrix, x_corr)

    # getSound(sub_wave_matrix,44100)
    # getSound(wave_matrix,44100)
    return np.argmin(x_corr)


def stereo_to_mono_wave(path):
    wave, fs = sf.read(path, dtype='float32')
    wave = np.delete(wave, 1, 1)
    return fs, wave
