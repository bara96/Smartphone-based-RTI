import constants as cst
import analysis
from FeatureMatcher import FeatureMatcher
from Utils import utilities as ut
import cv2
import matplotlib.pyplot as plt
import os
import numpy as np
from timeit import default_timer as timer


def Mouse_Event(event, x, y, flags, param):
    lx, ly = draw_light(x, y)

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if event == cv2.EVENT_LBUTTONDOWN:
        # apply relighting
        draw_relighting(lx, ly)


def draw_light(x, y):
    global light_pos_img

    img = light_pos_img.copy()
    cv2.circle(img, (x, y), 1, cst.COLOR_PURPLE, 5)
    cv2.imshow('Light Position', img)
    h, w = light_pos_img.shape
    lx = 2 * (x / w) - 1
    ly = 2 * (y / h) - 1
    lx = round(lx, 2)
    ly = round(ly, 2)
    print(lx, ly)

    return lx, ly


def draw_relighting(lx, ly):
    global roi_img
    global interpolation_intensities

    img = roi_img.copy()

    int_ly = round((1 + ly) / 2 * 100)
    int_lx = round((1 + lx) / 2 * 100)
    print(int_lx, int_ly)

    for y in range(cst.ROI_DIAMETER):
        for x in range(cst.ROI_DIAMETER):
            intensities = interpolation_intensities[y][x]
            # transform lx, ly values to interpolation_intensities coordinates

            intensity = intensities[int_ly][int_lx]
            img[y][x] = intensity
            # img[y][x] = get_pixel_intensity(roi_img[y][x], img_gray[y][x], intensity)

    cv2.imshow('Relighting', img)


def get_pixel_intensity(bgr, gray, intensity):
    B, G, R = bgr
    if intensity > 0:
        intensity = 0
    if intensity > 255:
        intensity = 255

    diff = abs(gray - intensity)
    if gray > intensity:
        return B - diff, G - diff, R - diff
    else:
        return B + diff, G + diff, R + diff


def compute(video_name='coin1', storage_filepath=None):
    """
    Main function
    :param video_name: name of the video to take
    :param storage_filepath: if None is set read results from default filepath, otherwise it must be a filepath to a valid results file
    interpolation_intensities: list of interpolated values foreach pixel
    """
    global roi_img
    global light_pos_img
    global interpolation_intensities

    results_filepath = "assets/interpolation_results_{}".format(video_name)

    if storage_filepath is not None:
        results_filepath = storage_filepath

    print("Reading interpolation values")
    interpolation_intensities = ut.read_from_file(results_filepath, compressed=False)

    # yi, xi = np.mgrid[-1:1:cst.INTERPOLATION_PARAM, -1:1:cst.INTERPOLATION_PARAM]

    default_frame_path = "assets/default_" + video_name + ".png"
    if not os.path.isfile(default_frame_path):
        default_frame_name = 'default_{}'.format(video_name)
        video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(video_name)
        default_frame_path = analysis.generate_video_default_frame(video_path=video_static_path,
                                                                   calibration_file_path=cst.INTRINSICS_STATIC_PATH,
                                                                   file_name=default_frame_name)

    print("Reading default frame")
    fm = FeatureMatcher()
    frame_default = cv2.imread(default_frame_path)
    static_shape_cnts, static_shape_points = fm.computeStaticShape(frame_default)
    roi_img = ut.get_ROI(frame_default, static_shape_points)
    roi_img = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)

    h, w = roi_img.shape
    h2, w2 = int(h / 2), int(w / 2)
    light_pos_img = np.zeros((h, w), np.uint8)
    cv2.line(light_pos_img, (0, h2), (w, h2), (255, 255, 255), 1)
    cv2.line(light_pos_img, (w2, 0), (w2, h), (255, 255, 255), 1)

    # Read input image, and create output image
    cv2.imshow('Relighting', roi_img)
    draw_light(w2, h2)

    # set Mouse Callback method
    cv2.setMouseCallback('Light Position', Mouse_Event)

    ut.console_log("Relightin On. \n", 'green')
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coin = 1
    storage_results_save = "assets/frames_results_coin{}".format(coin)

    start = timer()
    compute(video_name='coin1')
    print("Computation duration: {} s".format(round(timer() - start, 2)))
