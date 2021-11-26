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
            gray = ut.enchant_brightness_and_contrast(img)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = 255 - gray
            # apply morphology
            gray = ut.image_blur(gray, iterations=10)
            # gray = ut.enchant_morphological(gray, [cv2.MORPH_OPEN, cv2.MORPH_CLOSE], iterations=1)

            # find image edges
            canny = self._find_edges(gray)

            # refine all contours
            cnts = self._find_contours(canny)
            cv2.drawContours(canny, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)

            # draw only the longest contour
            canvas = np.zeros(gray.shape, np.uint8)  # create empty image from gray
            cnts = self._find_contours(canny, True)

            cv2.drawContours(canvas, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)

            # find corners
            corners = cv2.goodFeaturesToTrack(image=canvas,
                                              maxCorners=4,
                                              qualityLevel=0.1,
                                              minDistance=30,
                                              blockSize=20,
                                              useHarrisDetector=False)

            if corners is not None and len(corners) > 0:
                corners = np.int0(corners)
                # find_default_corner(img, corners)
                for corner in corners:
                    x, y = corner.ravel()
                    cv2.circle(img, (x, y), 1, (0, 0, 255), 10)
                    # check_neighbours(img, x, y)

            if show_params is True:
                cv2.imshow('canny', cv2.resize(canny, None, fx=0.6, fy=0.6))
                cv2.imshow("canvas", cv2.resize(canvas, None, fx=0.6, fy=0.6))
                cv2.imshow('results', cv2.resize(img, None, fx=0.6, fy=0.6))
                cv2.waitKey(0)
            else:
                print("No corners detected")

        return dataset

    @staticmethod
    def find_default_corner(img, corners):
        # search x and y bounds
        max_x = 0
        min_x = img.shape[1]
        max_y = 0
        min_y = img.shape[0]
        for corner in corners:
            x, y = corner.ravel()
            if x > max_x:
                max_x = x
            elif x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            elif y < min_y:
                min_y = y

        # search default point
        show_img = img.copy()
        default_corner = None
        min_distance = 1000
        x_median = round(max_x - (max_x - min_x) / 2)
        y_median = round(max_y - (max_y - min_y) / 2)
        for corner in corners:
            x, y = corner.ravel()
            if x < x_median:
                x_direction = 1
            else:
                x_direction = -1

            if y < y_median:
                y_direction = 1
            else:
                y_direction = -1

            distance = FeatureMatcher._search_white_border(img, x, y, x_direction, y_direction, show_img=show_img)
            print("\n")
            if distance is not False and distance < min_distance:
                default_corner = corner
                min_distance = distance

        cv2.imshow("test", cv2.resize(show_img, None, fx=0.6, fy=0.6))
        return default_corner

    @staticmethod
    def _search_white_border(img, x, y, x_direction, y_direction, limit=100, show_img=None):
        starting_x = x
        starting_y = y
        for i in range(1, limit):
            y += y_direction
            x += x_direction
            B1, _, _ = img[starting_y][x]
            B2, _, _ = img[y][starting_x]
            B3, _, _ = img[y][x]
            if show_img is not None:
                cv2.circle(show_img, (x, y), 1, (0, 255, 0), 10)
                cv2.circle(show_img, (starting_x, y), 1, (0, 255, 0), 10)
                cv2.circle(show_img, (x, starting_y), 1, (0, 255, 0), 10)
            if B1 > 50 or B2 > 50 or B3 > 50:
                return i
        return False

    @staticmethod
    def _find_edges(img):
        """
        Find image edges with Canny
        :param img:
        :return:
        """
        sigma = 0.4
        # compute the median of the single channel pixel intensities
        v = np.median(img)
        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        canny = cv2.Canny(img, lower, upper)
        # canny = cv2.Canny(gray, 120, 140)
        # canny = np.float32(canny)

        return canny

    @staticmethod
    def _find_contours(img, max_only=False):
        # thresh = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Find contours and sort for largest contour
        cnts, _ = cv2.findContours(img, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        max_perimeter = 0
        cnt = None
        perimeters = []
        if len(cnts) > 0:
            for c in cnts:
                p = cv2.arcLength(c, True)
                if p > max_perimeter:
                    max_perimeter = p
                    cnt = c
                perimeters.append(p)

        if max_only:
            return np.array(cnt)

        return cnts
