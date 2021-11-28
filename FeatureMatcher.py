import os
import cv2
import constants as cst
from matplotlib import pyplot as plt
import utilities as ut
import numpy as np


class FeatureMatcher:
    def __init__(self, frames_moving_folder_path, show_params=False):
        """
        Constructor
        :param frames_moving_folder_path: path to the moving frames folder
        """
        self.frames_moving_folder_path = frames_moving_folder_path
        if show_params is True:
            self.showParams(show_canny=True, show_rectangle_canvas=True, show_result=True, show_homography=True)
        else:
            self.showParams(show_canny=False, show_rectangle_canvas=False, show_result=False, show_homography=False)

    def showParams(self, show_canny=True, show_rectangle_canvas=True, show_result=True, show_homography=True):
        """
        Set show parameters
        :param show_canny: show canny detected edges
        :param show_rectangle_canvas: show detected rectangle canvas
        :param show_result: show result
        :param show_homography: show homography
        """
        self._show_canny = show_canny
        self._show_rectangle_canvas = show_rectangle_canvas
        self._show_corners = show_result
        self.show_homography = show_homography

    def extractFeatures(self):
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
        previous_second_corner = None
        for i in range(0, tot_frames):
            # Read the query image
            filename = self.frames_moving_folder_path + "/frame_{}.png".format(i)
            print("Frame n° ", i)
            img = cv2.imread(filename)
            height, width, _ = img.shape

            ''' Image Enchanting '''
            gray = ut.enchant_brightness_and_contrast(img)
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
            gray = 255 - gray
            gray = ut.image_blur(gray, iterations=5)
            # gray = ut.enchant_morphological(gray, [cv2.MORPH_CLOSE], iterations=1)

            ''' Image Refinement'''
            # find image edges
            canny = self._findEdges(gray)

            # refine all contours
            cnts = self._findContours(canny)
            cv2.drawContours(canny, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)

            # draw only the longest contour (bigger rectangle)
            rectangle_canvas = np.zeros(gray.shape, np.uint8)  # create empty image from gray
            cnts = self._findContours(canny, True, show_contours=False)

            cv2.drawContours(rectangle_canvas, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)

            ''' Corner Detection'''
            # find corners
            corners = cv2.goodFeaturesToTrack(image=rectangle_canvas,
                                              maxCorners=4,
                                              qualityLevel=0.1,
                                              minDistance=30,
                                              blockSize=20,
                                              useHarrisDetector=False)

            if corners is None or len(corners) != 4:
                ut.console_log("Error: Wrong corners detected", 'e')
                continue
            corners = np.int0(corners)

            # find the default corner
            default_corner = self.findDefaultCorner(img, corners)
            if default_corner is None:
                ut.console_log("Error: Default corner not found", 'e')
                continue

            distances_default = []
            distances_second = []
            for c in range(0, len(corners)):
                x, y = corners[c].ravel()
                # calculate for each corner the distance between default corner
                distance_default = ut.euclidean_distance(x, y, default_corner[0], default_corner[1])
                if distance_default > 0:
                    cv2.circle(img, (x, y), 1, cst.COLOR_RED, 10)
                    cv2.putText(img, str(distance_default), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, cst.COLOR_RED,
                                2, cv2.LINE_AA)
                    distances_default.append(dict(index=c, point=(x, y), distance=distance_default))
                    # calculate the distance between points and previous second-corner point
                    if previous_second_corner is not None:
                        distance_second = ut.euclidean_distance(x, y,
                                                                previous_second_corner[0],
                                                                previous_second_corner[1])
                        distances_second.append(dict(index=c, point=(x, y), distance=distance_second))

            # search for second-corner
            if previous_second_corner is not None:
                cv2.circle(img, previous_second_corner, 1, cst.COLOR_PURPLE, 10)
            second_corner = self.findSecondCorner(img, distances_default, distances_second)
            previous_second_corner = second_corner

            if self._show_canny is True:
                cv2.imshow('Canny', cv2.resize(canny, None, fx=0.6, fy=0.6))
            if self._show_rectangle_canvas is True:
                cv2.imshow("Rectangle Canvas", cv2.resize(rectangle_canvas, None, fx=0.6, fy=0.6))
            if self._show_corners is True:
                cv2.imshow('Corners', cv2.resize(img, None, fx=0.6, fy=0.6))
            cv2.waitKey(0)

        return dataset

    @staticmethod
    def findDefaultCorner(img, corners, show_point=True):
        """
        Find the default corner of the rectangle
        The default corner is the nearest to the circle
        :param img: OpenCv image
        :param corners: corners of the rectangle
        :param show_point: if True, show the point on image
        :return:
        """
        # search x and y bounds
        max_x = 0
        min_x = img.shape[1]
        max_y = 0
        min_y = img.shape[0]
        for corner in corners:
            x, y = corner.ravel()
            if x > max_x:
                max_x = x
            if x < min_x:
                min_x = x
            if y > max_y:
                max_y = y
            if y < min_y:
                min_y = y

        # search default point
        show_img = None  # img.copy()
        default_corner = None
        min_distance = 100
        x_median = round(max_x - (max_x - min_x) / 2)
        y_median = round(max_y - (max_y - min_y) / 2)
        for corner in corners:
            x, y = corner.ravel()
            distance = FeatureMatcher._searchWhiteBorder(img,
                                                         x_start=x,
                                                         y_start=y,
                                                         x_destination=x_median,
                                                         y_destination=y_median,
                                                         limit=min_distance,
                                                         show_img=show_img)

            if distance is not False and distance < min_distance:
                default_corner = (x, y)
                min_distance = distance

        if show_point:
            cv2.circle(img, default_corner, 1, cst.COLOR_BLUE, 10)

        return default_corner

    @staticmethod
    def _searchWhiteBorder(img, x_start, y_start, x_destination, y_destination, limit=1000, show_img=None):
        """
        Search for the white border aiming for center of the rectangle
        :param img: OpenCv image
        :param x_start:
        :param y_start:
        :param x_destination:
        :param y_destination:
        :param show_img: OpenCv image, if is set draw the search lines on the image
        :return:
        """
        # get line pixels points
        points = ut.bresenham_line((x_start, y_start), (x_destination, y_destination))
        for i in range(0, len(points)):
            x_prev, y_prev = points[i - 1]
            x, y = points[i]
            # stop if limit is reached
            if i > limit:
                return False

            if show_img is not None:
                cv2.circle(show_img, (x, y), 1, cst.COLOR_GREEN, 2)

            # check pixel intensity variation, i>10 to avoid pixels starting out of canvas
            diff_B, diff_G, diff_R = ut.get_pixel_variation(img[y][x], img[y_prev][x_prev])
            if i > 15 and (diff_B > 15 or diff_G > 15 or diff_R > 15):
                # print("diff: ", diff_B, diff_G, diff_R)
                return i
        return False

    @staticmethod
    def findSecondCorner(img, distances_default, distances_second, show_point=True):
        """
        Calculate second-corner point
        We will define a 'second' corner as a corner near to the default, and we will track it between frames
        :param img: OpenCv image
        :param distances_default: distances from the default_corner and the corners
        :param distances_second: distances from the previous_second and the corners
        :param show_point: if True, show the point on image
        :return:
        """

        distances_default = sorted(distances_default, key=lambda item: item['distance'])
        second_corner = distances_default[0].get('point')

        if len(distances_second) <= 0:
            if show_point:
                cv2.circle(img, second_corner, 1, cst.COLOR_GREEN, 10)
            return second_corner

        # the one nearest to the previous_second and that is not the farthest from default is the current second-corner
        distances_second = sorted(distances_second, key=lambda item: item['distance'])

        second_corner = distances_second[0].get('point')
        '''
        k = 0
        found = False
        while k < len(distances_second) or found is False:
            corner_idx = distances_second[k].get('index')
            if distances_default[-1].get('index') != corner_idx:
                second_corner = distances_second[k].get('point')
                found = True
            k += 1
        '''
        if show_point:
            cv2.circle(img, second_corner, 1, cst.COLOR_GREEN, 10)
        return second_corner

    @staticmethod
    def _findEdges(img):
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
    def _findContours(img, max_only=False, show_contours=False):
        """
        Find image contours
        :param img:
        :param max_only:
        :return:
        """
        # thresh = cv2.threshold(canvas, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # Find contours and sort for largest contour
        cnts, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

        max_perimeter = 0
        cnt = None
        perimeters = []
        cnts_approx = []
        i = 0
        if len(cnts) > 0:
            for c in cnts:
                peri = cv2.arcLength(c, closed=True)
                if show_contours and peri > 1000:
                    i += 1
                    test_img = np.zeros(img.shape, np.uint8)  # create empty image from gray
                    cv2.drawContours(test_img, [c], -1, (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.putText(test_img, "perimeter: {}" .format(peri), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255),
                                2, cv2.LINE_AA)
                    cv2.imshow("test{}".format(i), cv2.resize(test_img, None, fx=0.6, fy=0.6))
                if peri > max_perimeter:
                    max_perimeter = peri
                    cnt = [c]
                cnts_approx.append(c)
                perimeters.append(peri)

        if max_only:
            return np.array(cnt)

        return np.array(cnts_approx)
