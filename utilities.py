# Import required modules
# Utils misc functions
import cv2
import numpy as np
import os


def plot_waves(sub_wave_matrix, wave_matrix, x_corr):
    """
    Plot signal waves
    :param sub_wave_matrix:
    :param wave_matrix:
    :param x_corr:
    """
    import matplotlib.pyplot as plt

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


def undistort_image(image, matrix, distortion):
    """
    Un-distort the image
    :param image: OpenCv image
    :param matrix: intrinsics matrix
    :param distortion: intrinsics distortion
    :return:
    """
    # Compute the undistorted image
    h, w = image.shape[:2]
    # Compute the newcamera intrinsic matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 0)
    # Undistort image
    image_new = cv2.undistort(image, matrix, distortion, None, new_camera_matrix)
    # crop the image
    x, y, w, h = roi
    image_new = image_new[y:y + h, x:x + w]

    return image_new


def get_video_total_frames(video_path):
    """
    Return total n° of frames of a video
    :param video_path: path to the video
    :return:
    """
    video = cv2.VideoCapture(video_path)
    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    video.release()
    return tot_frames


def svg_to_png(image_path):
    """
    Convert svg to png
    :param image_path: path to the image
    :return:
    """
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF, renderPM

    image_path_new = image_path.replace('svg', 'png')

    drawing = svg2rlg(image_path)
    renderPM.drawToFile(drawing, image_path_new, fmt="PNG")

    return image_path_new


def enchant_morphological(image, params, iterations=1):
    """
    Enchantment of an image with morphological operations
    :param image: OpenCv image
    :param params: morphological operations to apply
    :param iterations: iterations to apply
    :return:
    """
    kernel = np.ones((5, 5), np.uint8)

    for type in params:
        if type == cv2.MORPH_ERODE:
            image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel, iterations=iterations)
        if type == cv2.MORPH_DILATE:
            image = cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel, iterations=iterations)
        if type == cv2.MORPH_OPEN:
            # erosion followed by dilation: removing noise
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
        if type == cv2.MORPH_CLOSE:
            # dilation followed by erosion: closing small holes inside the foreground objects
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        if type == cv2.MORPH_GRADIENT:
            # difference between dilation and erosion of an image
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel, iterations=iterations)
        if type == cv2.MORPH_TOPHAT:
            # difference between input image and opening of the image
            image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel, iterations=iterations)
        if type == cv2.MORPH_BLACKHAT:
            # difference between the closing of the input image and input image
            image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        if type == 'sharpen':
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

    return image


def enchant_CLAHE(image):
    """
    Contrast Limited Adaptive Histogram Equalization
    :param image: OpenCv image
    :return:
    """
    # -----Converting image to LAB Color model-----------------------------------
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # -----Splitting the LAB image to different channels-------------------------
    l, a, b = cv2.split(lab)

    # -----Applying CLAHE to L-channel-------------------------------------------
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # -----Merge the CLAHE enhanced L-channel with the a and b channel-----------
    limg = cv2.merge((cl, a, b))

    # -----Converting image from LAB Color model to RGB model--------------------
    return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)


def enchant_brightness_and_contrast(image, clip_hist_percent=1):
    """
    Automatic brightness and contrast optimization with optional histogram clipping
    :param image: OpenCv image
    :param clip_hist_percent:
    :return:
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Calculate grayscale histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)

    # Calculate cumulative distribution from the histogram
    accumulator = []
    accumulator.append(float(hist[0]))
    for index in range(1, hist_size):
        accumulator.append(accumulator[index - 1] + float(hist[index]))

    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0

    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1

    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1

    # Calculate alpha and beta values
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha

    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def cameraPoseFromHomography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H2)
    tnorm = (norm1 + norm2) / 2.0

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3, T])


def find_pose_from_homography(H, K):
    """
    function for pose prediction of the camera from the homography matrix, given the intrinsics

    :param H(np.array): size(3x3) homography matrix
    :param K(np.array): size(3x3) intrinsics of camera
    :Return t: size (3 x 1) vector of the translation of the transformation
    :Return R: size (3 x 3) matrix of the rotation of the transformation (orthogonal matrix)
    """

    # to disambiguate two rotation marices corresponding to the translation matrices (t and -t),
    # multiply H by the sign of the z-comp on the t-matrix to enforce the contraint that z-compoment of point
    # in-front must be positive and thus obtain a unique rotational matrix
    H = H * np.sign(H[2, 2])

    h1, h2, h3 = H[:, 0].reshape(-1, 1), H[:, 1].reshape(-1, 1), H[:, 2].reshape(-1, 1)

    R_ = np.hstack((h1, h2, np.cross(h1, h2, axis=0))).reshape(3, 3)

    U, S, V = np.linalg.svd(R_)

    R = U @ np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, np.linalg.det(U @ V.T)]]) @ V.T

    t = (h3 / np.linalg.norm(h1)).reshape(-1, 1)

    return R, t