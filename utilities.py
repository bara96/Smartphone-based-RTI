# Import required modules
# Utils misc functions
import constants as cst
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def console_log(message, status='e'):
    if status == 'e':
        print("\033[91m{}\033[0m".format(message))
    elif status == 'w':
        print("\033[93m{}\033[0m".format(message))


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
    :param data: Object to save
    :param filename: filename of the pickle file to save
    """
    import pickle
    with open(filename, "wb") as f:
        pickle.dump(data, f)


def read_from_file(filename):
    """
    Read data from pickle file
    :param filename: filename of the pickle file to read
    :return:
    """
    import pickle
    with open(filename, 'rb') as f:
        results = pickle.load(f)
    return results


def image_blur(image, iterations=1):
    """
    Blur image with GaussianBlur
    :param image: Opencv image
    :param iterations: n° of times to apply GaussianBlur
    :return:
    """
    for k in range(0, iterations):
        image = cv2.GaussianBlur(image, (5, 5), 0)
    return image


def image_fill(img, enlarge_percentage=1.5):
    """
    Enlarge an image with black
    :param img: OpenCv image
    :param enlarge_percentage: % of image to enlarge
    :return:
    """
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


def image_draw_circle(img, x, y, color=cst.COLOR_RED, radius=250, thickness=10):
    """
    Draw a point into an image
    :param thickness: circle line thickness
    :param radius: circle radius
    :param img: OpenCv image
    :param x: real world coordinate
    :param y: real world coordinate
    :param color: circle color
    :return:
    """
    max_y, max_x, _ = img.shape
    x, y = int(x), int(y)
    if x > max_x:
        x = max_x - 1
    if x < 0:
        x = 1

    if y > max_y:
        y = max_y - 1
    if y < 0:
        y = 1

    return cv2.circle(img, (x, y), radius=radius, color=color, thickness=thickness), x, y


def find_homography(src_pts, dst_pts):
    """
    Compute homography
    :param src_pts:
    :param dst_pts:
    :return:
    """
    src_pts = np.float32(src_pts).reshape(-1, 1, 2)
    dst_pts = np.float32(dst_pts).reshape(-1, 1, 2)

    matrix, mask = cv2.findHomography(src_pts, dst_pts, None, 6)
    if matrix is None:
        return None

    return matrix, mask


def find_camera_pose(src_shape_points, dst_shape_points, imageSize):
    """
    Find R and T from calibration
    :param src_shape_points: points from the world reference shape
    :param dst_shape_points: points from the secondary shape
    :param imageSize: size of te image
    :return:
    :return:
    R is rotation
    T is translation
    """

    points_3d = np.float32(
        [(src_shape_points[point][0], src_shape_points[point][1], 0) for point in
         range(0, len(src_shape_points))])
    points_2d = np.float32(
        [(dst_shape_points[point][0], dst_shape_points[point][1]) for point in
         range(0, len(dst_shape_points))])

    # perform a camera calibration to get R and T
    (ret, matrix, distortion, r_vecs, t_vecs) = cv2.calibrateCamera([points_3d], [points_2d],
                                                                    imageSize,
                                                                    None, None)
    R = cv2.Rodrigues(r_vecs[0])[0]
    T = t_vecs[0]

    return R, T


def find_pose_PNP(src_points, dst_points, calibration_file_path):
    """
    Solve PNP and use Rodrigues to find Camera world position
    :param src_points: keypoints of the first image
    :param dst_points: keypoints of the second image
    :param calibration_file_path: camera intrinsics path
    :return:
    """

    K, d = get_camera_intrinsics(calibration_file_path)

    src_points = np.float32([(src_points[point][0], src_points[point][1], 1) for point in range(0, len(src_points))])
    dst_points = np.float32([(dst_points[point][0], src_points[point][1]) for point in range(0, len(dst_points))])

    ret, rvecs, tvecs = cv2.solvePnP(src_points, dst_points, K, d)
    rotM = cv2.Rodrigues(rvecs)[0]
    camera_position = -np.matrix(rotM).T * np.matrix(tvecs)
    # print(camera_position)
    # camera_position = -rotM.transpose() * tvecs

    return camera_position


def get_pixel_variation(pixel1, pixel2):
    """
    Get color difference between two pixels
    :param pixel1:
    :param pixel2:
    :return:
    """
    if pixel1 is None or pixel2 is None:
        return 0, 0, 0
    B1, G1, R1 = int(pixel1[0]), int(pixel1[1]), int(pixel1[2])
    B2, G2, R2 = int(pixel2[0]), int(pixel2[1]), int(pixel2[2])
    return abs(B1 - B2), abs(G1 - G2), abs(R1 - R2)


def bresenham_line(start, end):
    """
    Bresenham's Line Algorithm
    Produces a list of tuples from start and end
    :param start:
    :param end:
    :return:
    """
    # Setup initial conditions
    x1, y1 = start
    x2, y2 = end
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1

    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points


def euclidean_distance(x1, y1, x2, y2):
    import math
    """
    Calculate euclidean distance between given points
    :param x1: 
    :param y1: 
    :param x2: 
    :param y2: 
    :return: 
    """
    # dist = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    dist = abs(x2 - x1) + abs(y2 - y1)
    return float(dist)


def interpolate_RBF(img):
    from PIL import Image
    from scipy.interpolate import Rbf

    img_data = np.asarray(img)

    # img = Rbf(img_data[:, 0], img_data[:, 1], img_data[:, 2])
    # val_ar = rbfi(img_data[:, 0], img_data[:, 1], img_data[:, 2])

    img_pil = Image.fromarray(np.uint8(img_data)).convert('RGB')
    img_pil.show()
