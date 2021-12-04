"""
Utilities methods for audio management
"""
import numpy as np
import matplotlib.pyplot as plt
import os


def plot_waves(sub_wave_matrix, wave_matrix, x_corr):
    """
    Plot signal waves
    :param sub_wave_matrix:
    :param wave_matrix:
    :param x_corr:
    """

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


def find_audio_correlation(sub_wave_matrix, wave_matrix, plot=False):
    """
    Get audio cross correlation
    :param sub_wave_matrix:
    :param wave_matrix:
    :param plot: if True plot the results
    :return:
    """
    from scipy import signal

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
    """
    Convert audio
    :param path:
    :return:
    """
    import soundfile as sf

    wave, fs = sf.read(path, dtype='float32')
    wave = np.delete(wave, 1, 1)
    return fs, wave


def get_audio_offset(video_static, video_moving):
    """
    Return video static offset, video moving offset
    :param video_static: first VideoFileClip
    :param video_moving: second VideoFileClip
    :return:
    """
    video1 = video_static
    swapped = False
    video2 = video_moving

    # swap the videos if the second one is greater
    if video1.duration > video2.duration:
        swapped = True
        video = video1
        video1 = video2
        video2 = video

    # extract audio from videos
    audio1 = video1.audio
    audio1_path = os.path.splitext(video1.filename)[0] + ".wav"
    audio1.write_audiofile(audio1_path)

    audio2 = video2.audio
    audio2_path = os.path.splitext(video2.filename)[0] + ".wav"
    audio2.write_audiofile(audio2_path)

    # get the video shift
    fs1, wave1 = stereo_to_mono_wave(audio1_path)
    fs2, wave2 = stereo_to_mono_wave(audio2_path)

    # calculate and round the shift
    offset = find_audio_correlation(wave2, wave1) / fs1

    print("Current offset: {} \n".format(offset))

    # return the offset
    if swapped:
        return offset, 0  # offset is on static video
    else:
        return 0, offset  # offset is on moving video
