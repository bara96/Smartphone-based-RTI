"""
Utilities methods for image/video management
"""
import constants as cst
import numpy as np
import cv2


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
