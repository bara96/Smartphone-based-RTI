"""
Main utilities methods
"""
# Utils misc functions
import constants as cst
import cv2
import numpy as np
import os


def console_log(message, color='red', newline=False):
    """
    Print a colored console message
    :param message:
    :param color:
    :param newline:
    """
    nl = ''
    if newline:
        nl = '\n'
    if color == 'red':
        print("{}\033[91m{}\033[0m".format(nl, message))
    elif color == 'green':
        print("{}\033[92m{}\033[0m".format(nl, message))
    elif color == 'yellow':
        print("{}\033[93m{}\033[0m".format(nl, message))
    elif color == 'blue':
        print("{}\033[94m{}\033[0m".format(nl, message))


def get_camera_intrinsics(calibration_file_path):
    import os
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


def write_on_file(data, filename, compressed=True):
    """
    Write data on pickle file
    :param data: Object to save
    :param filename: filename of the pickle file to save
    :param compressed: compress file
    """
    import bz2
    import pickle
    import _pickle as cPickle

    print("Writing on file {}".format(filename))

    if compressed:
        filename = filename + '.pbz2'
        with bz2.BZ2File(filename, "wb") as f:
            cPickle.dump(data, f)
    else:
        filename = filename + '.pickle'
        with open(filename, "wb") as f:
            pickle.dump(data, f)

    print("Saved file {}".format(filename))


def read_from_file(filename, compressed=True):
    """
    Read data from pickle file
    :param filename: filename of the pickle file to read
    :param compressed: read from compressed file
    :return:
    """
    import bz2
    import pickle
    import _pickle as cPickle

    if compressed:
        filename = filename + '.pbz2'
    else:
        filename = filename + '.pickle'

    if not os.path.isfile(filename):
        raise Exception('Storage file not found!')

    print("Reading from file {}".format(filename))

    if compressed:
        with bz2.BZ2File(filename, 'rb') as f:
            results = cPickle.load(f)
    else:
        with open(filename, "rb") as f:
            results = pickle.load(f)

    return results


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


def find_camera_pose(src_shape_points, dst_shape_points, refer_image, calibration_file_path=None):
    """
    Find R and T from calibration
    :param src_shape_points: points from the world reference shape
    :param dst_shape_points: points from the secondary shape
    :param refer_image: size of the image
    :param calibration_file_path: path to the intrinsics calibration file
    :return:
    R is rotation
    T is translation
    """

    refer_image = cv2.cvtColor(refer_image, cv2.COLOR_BGR2GRAY)
    image_size = refer_image.shape[::-1]
    # image_size = (refer_image.shape[0], refer_image.shape[1])

    if calibration_file_path is None:
        M, d = None, None
        z_axis = 0
        flags = None
    else:
        M, d = get_camera_intrinsics(calibration_file_path)
        z_axis = 1
        flags = cv2.CALIB_USE_INTRINSIC_GUESS

    points_3d = np.float32(
        [(src_shape_points[point][0], src_shape_points[point][1], z_axis) for point in
         range(0, len(src_shape_points))])
    points_2d = np.float32(
        [(dst_shape_points[point][0], dst_shape_points[point][1]) for point in
         range(0, len(dst_shape_points))])

    # perform a camera calibration to get R and T
    (ret, matrix, distortion, r_vecs, t_vecs) = cv2.calibrateCamera([points_3d],
                                                                    [points_2d],
                                                                    image_size,
                                                                    cameraMatrix=M,
                                                                    distCoeffs=d,
                                                                    flags=flags)
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


def get_corners_center(corners, height, width):
    """
    Get the center point between corners
    :param corners: corners points of the shape
    :param width: width of the OpenCv image
    :param height: height of the OpenCv image
    :rtype: object
    """
    # search x and y bounds
    max_x = 0
    min_x = width
    max_y = 0
    min_y = height
    for corner in corners:
        x, y = corner
        if x > max_x:
            max_x = x
        if x < min_x:
            min_x = x
        if y > max_y:
            max_y = y
        if y < min_y:
            min_y = y

    x_median = round(max_x - (max_x - min_x) / 2)
    y_median = round(max_y - (max_y - min_y) / 2)

    return x_median, y_median


def get_ROI(static_img, static_shape_points, grayscale=False, hsv=False, show_roi=False):
    """
    Extract Region Of Interest from an image with gray channel
    :param static_img: OpenCv image
    :param static_shape_points: points of the image
    :param grayscale: if True, return a greyscale image
    :param hsv: if True, return a hsv image
    :param show_roi: if True, show the extracted image
    :return:
    """
    roi_radius = round(cst.ROI_DIAMETER / 2)

    h, w, _ = static_img.shape
    corners = np.array(static_shape_points)
    x_center, y_center = get_corners_center(corners, h, w)
    x_min, x_max = x_center - roi_radius, x_center + roi_radius
    y_min, y_max = y_center - roi_radius, y_center + roi_radius

    if x_min < 0:
        x_min = 1
    if x_max > w:
        x_max = w - 1
    if y_min < 0:
        y_min = 1
    if y_max > h:
        y_max = h - 1

    roi_img = static_img[y_min:y_max, x_min:x_max].copy()

    if show_roi:
        cv2.imshow("roi_img", roi_img)

    if grayscale is True:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        return np.array(roi_img)

    if hsv is True:
        roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2HSV)
        return np.array(roi_img)

    return roi_img


def create_light_roi(frame_default, static_shape_points):
    roi_img = get_ROI(frame_default, static_shape_points)
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
    h, w = roi_img.shape
    h2, w2 = int(h / 2), int(w / 2)
    light_pos_img = np.zeros((h, w), np.uint8)
    cv2.line(light_pos_img, (0, h2), (w, h2), (255, 255, 255), 1)
    cv2.line(light_pos_img, (w2, 0), (w2, h), (255, 255, 255), 1)
    light_pos_img = cv2.cvtColor(light_pos_img, cv2.COLOR_GRAY2BGR)
    return light_pos_img


def draw_light_roi_position(given_x, given_y, shape, to_light_vector=False):
    """
    get x,y or lx,ly coordinates for light ROI
    :param given_x:
    :param given_y:
    :param shape:
    :param to_light_vector: if True get lx,ly from x,y otherwise get x,y from lx,ly
    :return:
    """
    h, w = shape

    if to_light_vector:
        # get lx,ly from x,y
        lx = round(2 * (given_x / w) - 1, 2)
        ly = round(2 * (given_y / h) - 1, 2)
        if lx >= 0.99:
            lx = 0.98
        if ly >= 0.99:
            ly = 0.98
        return lx, ly
    else:
        # get x,y from lx,ly
        x = int(2 * (1 + given_x) * 100)
        y = int(2 * (1 + given_y) * 100)
        return x, y
