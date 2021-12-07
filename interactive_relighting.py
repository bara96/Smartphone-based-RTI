import cv2
import constants as cst
import analysis
from FeatureMatcher import FeatureMatcher
from Utils import utilities as ut
import os
import numpy as np


def Mouse_Event(event, x, y, flags, param):
    global roi_img
    global results_interpolation

    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        # apply relighting
        for y in range(cst.ROI_DIAMETER):
            for x in range(cst.ROI_DIAMETER):
                intensities = results_interpolation[y][x]
                # lx = 0, ly = 0
                intensity = intensities[0][0]
                print(intensity)

        cv2.imshow('Relighting', roi_img)


def compute(video_name='coin1', storage_filepath=None):
    """
    Main function
    :param video_name: name of the video to take
    """
    global roi_img
    global results_interpolation

    results_filepath = "assets/interpolation_results_{}.pickle".format(video_name)

    if storage_filepath is not None:
        results_filepath = storage_filepath
    if not os.path.isfile(results_filepath):
        raise Exception('Storage results file not found!')

    print("Reading interpolation values")
    results_interpolation = ut.read_from_file(results_filepath)

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

    ut.console_log("Relightin On. \n", 'green')
    # Read input image, and create output image
    cv2.imshow('Relighting', roi_img)

    # set Mouse Callback method
    cv2.setMouseCallback('Relighting', Mouse_Event)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coin = 1
    storage_results_save = "assets/frames_results_coin{}_save.pickle".format(coin)
    compute(video_name='coin1')
