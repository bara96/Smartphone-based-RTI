import os
import cv2
import constants as cst
from matplotlib import pyplot as plt
import utilities as ut
import numpy as np


class FeatureMatcher:
    def __init__(self, frames_moving_folder_path):
        """
        Constructor
        :param frames_moving_folder_path: path to the moving frames folder
        """
        self.frames_moving_folder_path = frames_moving_folder_path

    def extractFeatures(self, show_params=True):
        """
        Feature matching and homography check
        :param show_params: if True show all results
        :return:
        """

        if not os.path.isdir(self.frames_moving_folder_path):
            raise Exception('Moving folder not found!')

        # read frames from folders
        list_moving = os.listdir(self.frames_moving_folder_path)
        tot_frames = len(list_moving)

        dataset = []
        for i in range(0, tot_frames):
            # Read the query image
            filename = self.frames_moving_folder_path + "/frame_{}.png".format(i)
            img = cv2.imread(filename)
            img = ut.enchant_brightness_and_contrast(img)
            img = ut.image_blur(img, iterations=10)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            canny = cv2.Canny(gray, 120, 255, 1)
            corners = cv2.goodFeaturesToTrack(image=gray,
                                              maxCorners=100,
                                              qualityLevel=0.2,
                                              minDistance=10)
            corners = np.int0(corners)
            if len(corners) > 0:
                for corner  in corners:
                    x, y = corner.ravel()
                    cv2.circle(img, (x, y), 1, (0,0,255), 10)

            if show_params is True:
                #cv2.imshow('canny', cv2.resize(canny, None, fx=0.6, fy=0.6))
                cv2.imshow('results', cv2.resize(img, None, fx=0.6, fy=0.6))
                cv2.waitKey(0)

        return dataset
