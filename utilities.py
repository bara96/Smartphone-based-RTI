# Import required modules
import cv2
import numpy as np
import constants as cst


# plot signal waves
def plot_waves(sub_wave_matrix, wave_matrix, x_corr):
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


# get audio cross correlation
def find_audio_correlation(sub_wave_matrix, wave_matrix, plot=False):
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


# convert audio
def stereo_to_mono_wave(path):
    import soundfile as sf

    wave, fs = sf.read(path, dtype='float32')
    wave = np.delete(wave, 1, 1)
    return fs, wave


# undistort the image
# image: cv2 image Object
# matrix: intrinsics matrix
# distortion: intrinsics distortion
def undistort_image(image, matrix, distortion):
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


# return total n° of frames of a video
def get_video_total_frames(video_path):
    video = cv2.VideoCapture(video_path)
    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    video.release()
    return tot_frames


# convert svg to png
def svg_to_png(image_path):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF, renderPM

    image_path_new = image_path.replace('svg', 'png')

    drawing = svg2rlg(image_path)
    renderPM.drawToFile(drawing, image_path_new, fmt="PNG")

    return image_path_new


# enchantment of an image with morphological operations
def image_enchantment(image, params):
    kernel = np.ones((5, 5), np.uint8)

    for type in params:
        if type == 'erode':
            image = cv2.morphologyEx(image, cv2.MORPH_ERODE, kernel)
        if type == 'dilation':
            cv2.morphologyEx(image, cv2.MORPH_DILATE, kernel)
        if type == 'opening':
            # erosion followed by dilation: removing noise
            image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        if type == 'closing':
            # dilation followed by erosion: closing small holes inside the foreground objects
            image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        if type == 'gradient':
            # difference between dilation and erosion of an image
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        if type == 'tophat':
            # difference between input image and opening of the image
            image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        if type == 'blackhat':
            # difference between the closing of the input image and input image
            image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        if type == 'sharpen':
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

    return image


# find homography matrix and do perspective transform
def homography_transformation(refer_image, refer_features, transform_image, transform_features, matches,
                              show_images=True, save_as=None):
    import os

    kp_refer_image, desc_refer_image = refer_features[0], refer_features[1]
    kp_transform_image, desc_transform_image = transform_features[0], transform_features[1]

    refer_pts = np.float32([kp_transform_image[m.trainIdx]
                           .pt for m in matches]).reshape(-1, 1, 2)

    transform_pts = np.float32([kp_refer_image[m.queryIdx]
                               .pt for m in matches]).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(refer_pts, transform_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Warp query image to train image based on homography
    im_out = cv2.warpPerspective(transform_image, matrix, (refer_image.shape[1], refer_image.shape[0]))

    if show_images:
        cv2.imshow("Transformed", im_out)
        # cv2.waitKey(0)
    if save_as is not None:
        if not os.path.isdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH):
            os.mkdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH)
        cv2.imwrite(cst.TRANSFORMATION_RESULTS_FOLDER_PATH + '/' + save_as, im_out)

    return matrix


# Contrast Limited Adaptive Histogram Equalization
def CLAHE(image):
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


# Automatic brightness and contrast optimization with optional histogram clipping
def enchant_brightness_and_contrast(image, clip_hist_percent=1):
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
