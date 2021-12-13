import constants as cst
import analysis
from FeatureMatcher import FeatureMatcher
from Utils import utilities as ut
import os
import cv2
import numpy as np
from timeit import default_timer as timer


def relighting_event(event, x, y, flags, param):
    """
    Relighting event (mouse click)
    :param event:
    :param x:
    :param y:
    :param flags:
    :param param:
    """
    global interpolation_results
    lx, ly = draw_light(x, y, show_coordinates=True)

    # apply relighting
    int_ly = round((1 + ly) / 2 * 100)
    int_lx = round((1 + lx) / 2 * 100)
    #print("Cursor: ", int_ly, int_lx)

    img = np.array(interpolation_results[int_ly][int_lx], dtype=np.uint8)
    cv2.imshow('Relighting', img)


def draw_light(x, y, show_coordinates=False):
    """
    Draw light position
    :param x: x coordinate
    :param y: y coordinate
    :param show_coordinates: if True, show curso coordinates
    :return:
    """
    global light_pos_img

    img = light_pos_img.copy()
    cv2.circle(img, (x, y), 1, cst.COLOR_WHITE, 5)
    cv2.imshow('Light Position', img)
    h, w = light_pos_img.shape
    lx, ly = ut.draw_light_roi_position(x, y, (h, w), to_light_vector=True)

    if show_coordinates:
        print(lx, ly)

    return lx, ly


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
    interpolation_results: list of interpolated values foreach pixel
    """
    global light_pos_img
    global interpolation_results

    results_filepath = "assets/interpolation_results_{}".format(video_name)

    if storage_filepath is not None:
        results_filepath = storage_filepath

    print("Reading interpolation results")
    interpolation_results = ut.read_from_file(results_filepath, compressed=False)

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

    light_pos_img = ut.create_light_roi(frame_default, static_shape_points)
    h2, w2 = int(light_pos_img.shape[0]/2), int(light_pos_img.shape[1]/2)

    # Read input image, and create output image
    cv2.imshow('Relighting', roi_img)
    draw_light(w2, h2)

    # set Mouse Callback method
    cv2.setMouseCallback('Light Position', relighting_event)

    ut.console_log("Relightin On. \n", 'green')
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coin = 1
    storage_results_save = "assets/frames_results_coin{}".format(coin)

    start = timer()
    compute(video_name='coin1')
    print("Computation duration: {} s".format(round(timer() - start, 2)))
