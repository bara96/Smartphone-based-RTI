import cv2
import constants as cst
import utilities as ut
import os
import numpy as np

def Mouse_Event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_new_name = analysis_results[0]['trainImage']
        img_new = cv2.imread(img_new_name)
        img_new = cv2.resize(img_new, (600, 800))
        # apply histogram equalization
        cv2.imshow('Image', img_new)


def compute(video_name='coin1'):
    global img
    global analysis_results
    """
    Main function
    :param video_name: name of the video to take
    """
    # read results from file
    file_path = "assets/results_{}.pickle".format(video_name)
    if not os.path.isfile(file_path):
        raise Exception('Results file not found!')
    analysis_results = ut.read_from_file(file_path)

    default_frame_path = cst.FRAMES_FOLDER_PATH + "/default_" + video_name + ".png"
    image_default = cv2.imread(default_frame_path)

    # Read input image, and create output image
    img = cv2.resize(image_default, (600, 800))
    cv2.imshow('Image', img)

    # set Mouse Callback method
    cv2.setMouseCallback('Image', Mouse_Event)
    cv2.waitKey(0)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute()
