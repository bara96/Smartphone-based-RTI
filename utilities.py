# Import required modules
# Utils misc functions
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


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


def get_camera_intrinsics(calibration_file_path):
    """
    Get camera intrinsic matrix and distorsion
    :param calibration_file_path: file path to intrinsics file
    """
    if not os.path.isfile(calibration_file_path):
        raise Exception('intrinsics file not found!')
    else:
        # Read intrinsics to file
        Kfile = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
        matrix = Kfile.getNode("K").mat()
        distortion = Kfile.getNode("distortion").mat()

    return matrix, distortion


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


def write_on_file(data, filename):
    """
    Write data on pickle file
    :param data:
    :param filename:
    """
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_from_file(filename):
    """
    Read data from pickle file
    :param filename:
    :return:
    """
    import pickle
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def image_fill(img, enlarge_percentage=1.5):
    # Getting the bigger side of the image
    s = max(img.shape[0:2])
    s = round(s * enlarge_percentage)

    # Creating a dark square with NUMPY
    img_new = np.zeros((s, s, 3), np.uint8)

    # Getting the centering position
    ax, ay = (s - img.shape[1]) // 2, (s - img.shape[0]) // 2

    # Pasting the 'image' in a centering position
    img_new[ay:img.shape[0] + ay, ax:ax + img.shape[1]] = img

    return img_new


def image_draw_point(img, x, y, color=(0, 0, 255)):
    max_y, max_x, _ = img.shape
    x, y = abs(int(x)), abs(int(y))
    if x > max_x:
        x = max_x - 1
    if y > max_y:
        y = max_y - 1

    return cv2.circle(img, (x, y), radius=0, color=color, thickness=30), x, y


def find_pose_from_homography(H, K, img, show_position=True):
    """
    H is the homography matrix
    K is the camera calibration matrix
    T is translation
    R is rotation
    """

    H = H.T
    h1 = H[0]
    h2 = H[1]
    h3 = H[2]
    K_inv = np.linalg.inv(K)
    L = 1 / np.linalg.norm(np.dot(K_inv, h1))
    r1 = L * np.dot(K_inv, h1)
    r2 = L * np.dot(K_inv, h2)
    r3 = np.cross(r1, r2)
    T = L * (K_inv @ h3.reshape(3, 1))
    R = np.array([[r1], [r2], [r3]])
    R = np.reshape(R, (3, 3))

    # debug code
    vector = -(R.transpose() * T)

    x, y = vector[0][0], vector[0][1]       # red
    x1, y1 = vector[1][0], vector[1][1]     # yellow
    x2, y2 = vector[2][0], vector[2][1]     # purple

    # x,y,z = np.dot(-np.transpose(R),T)

    train_img_new = img.copy()
    x_scale, y_scale = 0.4, 0.4

    train_img_new, x, y = image_draw_point(train_img_new, x, y, (0, 0, 255))
    train_img_new, x1, y1 = image_draw_point(train_img_new, x1, y1, (0, 255, 255))
    train_img_new, x2, y2 = image_draw_point(train_img_new, x2, y2, (255, 0, 255))
    train_img_new = cv2.resize(train_img_new, None, fx=x_scale, fy=y_scale)

    # print("x: ", x, "y: ", y)
    print("x: ", round(x * x_scale), "y: ", round(y * y_scale))  # scaled values
    print("x1: ", round(x1 * x_scale), "y1: ", round(y1 * y_scale))  # scaled values
    print("x2: ", round(x2 * x_scale), "y2: ", round(y2 * y_scale))  # scaled values

    if show_position:
        cv2.imshow("Camera Position", train_img_new)

    return R, T


def find_pose_from_pnp(trainKeypoints, queryKeypoints, matches, K, d, img, show_position=True):
    train_pts = np.float32([np.append(trainKeypoints[m.trainIdx].pt, 1.) for m in matches])
    query_pts = np.float32([queryKeypoints[m.queryIdx].pt for m in matches])

    ret, rvecs, tvecs = cv2.solvePnP(train_pts, query_pts, K, d)
    rotM = cv2.Rodrigues(rvecs)[0]
    cameraPosition = -np.matrix(rotM).T * np.matrix(tvecs)
    # cameraPosition = -(rotM.transpose() * tvecs)

    train_img_new = img.copy()
    train_img_new, x, y = image_draw_point(train_img_new, cameraPosition[0][0], cameraPosition[1][0], (0, 0, 255))
    if show_position:
        cv2.imshow("Camera Position", cv2.resize(train_img_new, None, fx=0.4, fy=0.4))

    return train_img_new


def extract_pixel_intensity(img):
    imgWidth, imgHeight = img.size
    img = img.convert("RGBA")
    imgdata = img.getdata()

    x_pos = 0
    y_pos = 1
    pixel_value = []
    x = []
    y = []
    for item in imgdata:
        if (x_pos) == imgWidth:
            x_pos = 1
            y_pos += 1
        else:
            x_pos += 1

        if item[3] != 0:
            pixel_value.append(item[2])
            x.append(x_pos)
            y.append(y_pos)

    pixel_value, x, y = zip(*sorted(zip(pixel_value, x, y), reverse=True))
    return pixel_value, x, y


def interpolate_RBF(img):
    from PIL import Image
    from scipy.interpolate import Rbf

    img_data = np.asarray(img)

    # img = Rbf(img_data[:, 0], img_data[:, 1], img_data[:, 2])
    # val_ar = rbfi(img_data[:, 0], img_data[:, 1], img_data[:, 2])

    img_pil = Image.fromarray(np.uint8(img_data)).convert('RGB')
    img_pil.show()