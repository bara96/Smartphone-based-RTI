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
                                              maxCorners=500,
                                              qualityLevel=0.2,
                                              minDistance=10,
                                              useHarrisDetector=False)


            if corners is not None:
                corners = np.int0(corners)
                corners = corners.reshape(len(corners), 2)
                corners = sorted(corners, key=lambda tup: tup[0])  # order by x
                if len(corners) > 0:
                    for corner in corners:
                        x, y = corner.ravel()
                        cv2.circle(img, (x, y), 1, (0, 0, 255), 10)
                        check_neighbours(img, x, y)


                if show_params is True:
                    cv2.imshow('canny', cv2.resize(canny, None, fx=0.6, fy=0.6))
                    cv2.imshow('results', cv2.resize(img, None, fx=0.6, fy=0.6))
                    cv2.waitKey(0)
            else:
                print("No corners detected")

        return dataset


def check_neighbours(img, x, y):
    h, w, _ = img.shape

    d=1
    if x-d < 0 or y-d < 0 or x+d > w or y+d > h:
        print("Neighbours out of bound")
        return False

    cell_neighbors(img, x, y, 10)


def cell_neighbors(img, x, y, d=1):
    neighbours = []
    test = img.copy()
    for i in range(-d, d+1):
        pixels = []
        for k in range(-d, d+1):
            pixel_B, pixel_G, pixel_R = img[y+i][x+k]
            cv2.circle(test, (x+k, y+i), 1, (0, 0, 255), 1)
            pixels.append([pixel_B, pixel_G, pixel_R])
        neighbours.append(pixels)

    neighbours = np.array(neighbours)
    print(neighbours.shape)
    print(neighbours, "\n")
    cv2.imshow('test', cv2.resize(test, None, fx=0.6, fy=0.6))
    cv2.waitKey(0)
    return neighbours