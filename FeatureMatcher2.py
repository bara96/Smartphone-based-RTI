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
            print("Frame n° ", i)
            img = cv2.imread(filename)
            h, w, _ = img.shape

            ''' Image Enchanting Phase '''
            img = ut.enchant_brightness_and_contrast(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gray = 255 - gray
            # apply morphology
            gray = ut.image_blur(gray, iterations=10)
            gray = ut.enchant_morphological(gray, [cv2.MORPH_OPEN, cv2.MORPH_CLOSE], iterations=1)

            ''' Edge Detector'''
            sigma = 0.4
            # compute the median of the single channel pixel intensities
            v = np.median(gray)
            # apply automatic Canny edge detection using the computed median
            lower = int(max(0, (1.0 - sigma) * v))
            upper = int(min(255, (1.0 + sigma) * v))
            canny = cv2.Canny(gray, lower, upper)
            # canny = cv2.Canny(gray, 120, 140)
            corners = cv2.goodFeaturesToTrack(image=canny,
                                              maxCorners=100,
                                              qualityLevel=0.2,
                                              minDistance=10)

            top_left = (w, h)
            top_right = (0, h)
            bottom_left = (w, 0)
            bottom_right = (0, 0)
            if corners is not None:
                corners = np.int0(corners)
                if len(corners) > 0:
                    for corner in corners:
                        x, y = corner.ravel()
                        if x < top_left[0] and y < top_left[1]:
                            top_left = (x, y)
                        if x > top_right[0] and y < top_right[1]:
                            top_right = (x, y)
                        if x < bottom_left[0] and y > bottom_left[1]:
                            bottom_left = (x, y)
                        if x > bottom_right[0] and y > bottom_right[1]:
                            bottom_right = (x, y)
                        # cv2.circle(img, (x, y), 1, (0, 0, 255), 10)

                    cv2.circle(img, top_left, 1, (255, 0, 0), 10)
                    cv2.circle(img, top_right, 1, (0, 255, 0), 10)
                    cv2.circle(img, bottom_left, 1, (0, 255, 255), 10)
                    cv2.circle(img, bottom_right, 1, (255, 0, 255), 10)

                    print("top_left: ", ut.printCoordinates(top_left, 0.6))           # blue
                    print("top_right: ", ut.printCoordinates(top_right, 0.6))         # green
                    print("bottom_left: ", ut.printCoordinates(bottom_left, 0.6))     # yellow
                    print("bottom_right: ", ut.printCoordinates(bottom_right, 0.6))   # purple

                if show_params is True:
                    cv2.imshow('canny', cv2.resize(canny, None, fx=0.6, fy=0.6))
                    cv2.imshow('results', cv2.resize(img, None, fx=0.6, fy=0.6))
                    cv2.waitKey(0)
            else:
                print("No corners detected")

        return dataset
