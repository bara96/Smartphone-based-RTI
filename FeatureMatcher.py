import os
import cv2
import constants as cst
from matplotlib import pyplot as plt
import utilities as ut
import numpy as np


class FeatureMatcher:
    def __init__(self, show_params=False):
        """
        Constructor
        """
        self._previous_default_corner = None
        self._previous_second_corner = None
        self._previous_third_corner = None
        if show_params is True:
            self.setShowParams(show_static_frame=True, show_moving_frame=True, show_rectangle_canvas=True,
                               show_corners=True, show_previous_corners=True, show_homography=True,
                               show_light_direction=True)
        else:
            self.setShowParams(show_static_frame=True, show_moving_frame=True, show_rectangle_canvas=False,
                               show_corners=False, show_previous_corners=False, show_homography=False,
                               show_light_direction=False)

    def setShowParams(self, show_static_frame=True, show_moving_frame=True, show_rectangle_canvas=False,
                      show_corners=False, show_previous_corners=False, show_homography=False, show_light_direction=False):
        """
        Set show parameters
        :param show_static_frame: show default rectangle shape
        :param show_moving_frame: show default rectangle shape
        :param show_rectangle_canvas: show detected rectangles canvas on both images
        :param show_corners: show detected corners on both images
        :param show_previous_corners: show previous detected corners
        :param show_homography: show homography between static and moving frames
        :param show_light_direction: show light direction
        """
        self._show_static_frame = show_static_frame
        self._show_moving_frame = show_moving_frame
        self._show_rectangles_canvas = show_rectangle_canvas
        self._show_corners = show_corners
        self._show_previous_corners = show_previous_corners
        self._show_homography = show_homography
        self._show_light_direction = show_light_direction

    def resetPreviousCorners(self):
        """
        Reset previous corners points
        """
        self._previous_third_corner = None
        self._previous_second_corner = None

    def extractFeatures(self, moving_img, static_img, static_shape_points, wait_key=False):
        """
        Feature matching and homography check of given image
        :param moving_img: OpenCv image
        :param static_img: OpenCv image: default rectangle
        :param static_shape_points: points of the default rectangle to use into homography check
        :param wait_key: specify if wait to user input or not when showing frames
        :return:
        """

        if wait_key is False:
            wait_key = 1
        else:
            wait_key = 0

        data = []

        height, width, _ = moving_img.shape

        ''' Image Enchanting '''
        gray = ut.enchant_brightness_and_contrast(moving_img)
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
        if cnts is None:
            ut.console_log("Error Moving: No contours detected", 'e')
            return False

        cv2.drawContours(rectangle_canvas, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)
        if self._show_rectangles_canvas:
            cv2.drawContours(moving_img, cnts, -1, cst.COLOR_RED, 3, cv2.LINE_AA)

        ''' Corner Detection'''
        # find corners
        corners = cv2.goodFeaturesToTrack(image=rectangle_canvas,
                                          maxCorners=4,
                                          qualityLevel=0.1,
                                          minDistance=30,
                                          blockSize=20,
                                          useHarrisDetector=False)

        if corners is None or len(corners) != 4:
            ut.console_log("Error Moving: Wrong corners detected", 'e')
            return False
        corners = np.int0(corners)
        corners = corners.reshape((len(corners), 2))

        # find the default corner
        default_corner = None
        if self._previous_default_corner is None:
            # if there isn't a previous point calculate by searching the white circle
            default_corner = self._findDefaultCorner(moving_img, corners)
        else:
            # if there is a previous point calculate by distance
            min_distance = width
            for corner in corners:
                x, y = corner
                # cv2.putText(img, "{}  {}".format(x, y), (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
                # calculate for each corner the distance between default corner
                distance = ut.euclidean_distance(x, y, self._previous_default_corner[0],
                                                 self._previous_default_corner[1])
                if distance < min_distance:
                    default_corner = (x, y)
                    min_distance = distance
        if default_corner is None:
            ut.console_log("Error Moving: Default corner not found", 'e')
            return False
        self._previous_default_corner = default_corner

        if self._show_previous_corners:
            if self._previous_default_corner is not None:
                cv2.circle(moving_img, self._previous_default_corner, 1, cst.COLOR_PURPLE, 20)
            if self._previous_second_corner is not None:
                cv2.circle(moving_img, self._previous_second_corner, 1, cst.COLOR_PURPLE, 20)
            if self._previous_third_corner is not None:
                cv2.circle(moving_img, self._previous_third_corner, 1, cst.COLOR_PURPLE, 20)

        # detect and track rectangle corners given the default corner
        second_corner, third_corner, fourth_corner = self._findCorners(corners, default_corner)
        if self._show_corners is True:
            cv2.circle(moving_img, default_corner, 1, cst.COLOR_BLUE, 20)
            cv2.circle(moving_img, second_corner, 1, cst.COLOR_GREEN, 20)
            cv2.circle(moving_img, third_corner, 1, cst.COLOR_YELLOW, 20)
            cv2.circle(moving_img, fourth_corner, 1, cst.COLOR_ORANGE, 20)

        ''' Homography '''
        # find homography between moving and world
        moving_shape_points = (default_corner, second_corner, third_corner, fourth_corner)
        matrix, _ = ut.find_homography(moving_shape_points, static_shape_points)
        if matrix is not None:
            self._getStaticPixelsIntensity(static_img, static_shape_points)

            if self._show_homography:
                img_homography = cv2.warpPerspective(moving_img, matrix,
                                                     (static_img.shape[1], static_img.shape[0]))
                cv2.imshow("Homography", cv2.resize(img_homography, None, fx=0.4, fy=0.4))

            camera_position = ut.find_pose_PNP(static_shape_points, moving_shape_points, cst.INTRINSICS_MOVING_PATH)
            if self._show_light_direction:
                ut.image_draw_circle(static_img, camera_position[0], camera_position[1], cst.COLOR_RED)

        if self._show_static_frame:
            cv2.imshow('Static Camera', cv2.resize(static_img, None, fx=0.4, fy=0.4))
        if self._show_moving_frame:
            cv2.imshow('Moving Camera', cv2.resize(moving_img, None, fx=0.5, fy=0.5))
        cv2.waitKey(wait_key)

        return data

    def computeStaticShape(self, img):
        """
        Compute the static shape rectangle, return the 4 points
        :param img: OpenCv image
        """
        height, width, _ = img.shape

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = 255 - gray
        gray = ut.image_blur(gray, iterations=5)

        ''' Image Refinement'''
        # find image edges
        canny = FeatureMatcher._findEdges(gray)

        # refine all contours
        cnts = self._findContours(canny)
        cv2.drawContours(canny, cnts, -1, (255, 255, 255), 1, cv2.LINE_AA)

        # draw only the longest contour (bigger rectangle)
        rectangle_canvas = np.zeros(gray.shape, np.uint8)  # create empty image from gray
        cnts = self._findContours(canny, True, show_contours=False)
        if cnts is None:
            ut.console_log("Error Static: No contours detected", 'e')
            return False

        cv2.drawContours(rectangle_canvas, cnts, -1, (255, 255, 255), 3, cv2.LINE_AA)
        if self._show_rectangles_canvas:
            cv2.drawContours(img, cnts, -1, cst.COLOR_RED, 3, cv2.LINE_AA)

        ''' Corner Detection'''
        # find corners
        corners = cv2.goodFeaturesToTrack(image=rectangle_canvas,
                                          maxCorners=4,
                                          qualityLevel=0.1,
                                          minDistance=30,
                                          blockSize=20,
                                          useHarrisDetector=False)

        if corners is None or len(corners) != 4:
            ut.console_log("Error Static: Wrong corners detected", 'e')
            return img, None
        corners = np.int0(corners)
        corners = corners.reshape((len(corners), 2))

        default_corner = (width, 0)
        # set an acceptance threshold in order to spot corners
        corner_threshold = 10
        for corner in corners:
            x, y = corner
            if x < (default_corner[0] + corner_threshold) and y > (default_corner[1] - corner_threshold):
                default_corner = (x, y)

        # search for pre-defined corners
        second_corner = None
        third_corner = None
        fourth_corner = None
        for corner in corners:
            x, y = corner.ravel()
            # cv2.putText(img, "{}  {}".format(x, y), (x - 100, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3, cv2.LINE_AA)
            if x == default_corner[0] and y == default_corner[1]:
                continue
            if x < (default_corner[0] + corner_threshold):
                second_corner = (x, y)
            elif y > default_corner[1] - corner_threshold:
                third_corner = (x, y)
            else:
                fourth_corner = (x, y)

        if second_corner is None or third_corner is None or fourth_corner is None:
            ut.console_log("Error Static: Wrong corners detected", 'e')
            return img, None

        if self._show_corners is True:
            cv2.circle(img, default_corner, 1, cst.COLOR_BLUE, 20)
            cv2.circle(img, second_corner, 1, cst.COLOR_GREEN, 20)
            cv2.circle(img, third_corner, 1, cst.COLOR_YELLOW, 20)
            cv2.circle(img, fourth_corner, 1, cst.COLOR_ORANGE, 20)

        return img, (default_corner, second_corner, third_corner, fourth_corner)

    def _findCorners(self, corners, default_corner):
        """
        Find rectangle corners given the default
        :param corners:
        :param default_corner:
        :return:
        """
        distances_default = []
        distances_second = []
        distances_third = []
        for c in range(0, len(corners)):
            x, y = corners[c].ravel()
            # calculate for each corner the distance between default corner
            distance_default = ut.euclidean_distance(x, y, default_corner[0], default_corner[1])
            if distance_default > 0:
                # cv2.circle(img, (x, y), 1, cst.COLOR_RED, 10)
                # cv2.putText(img, str(distance_default), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, cst.COLOR_RED, 2, cv2.LINE_AA)
                distances_default.append(dict(index=c, point=(x, y), distance=distance_default))
                # calculate the distance between points and previous second-corner point
                if self._previous_second_corner is not None:
                    distance_second = ut.euclidean_distance(x, y,
                                                            self._previous_second_corner[0],
                                                            self._previous_second_corner[1])
                    distances_second.append(dict(index=c, point=(x, y), distance=distance_second))
                if self._previous_third_corner is not None:
                    distance_third = ut.euclidean_distance(x, y,
                                                           self._previous_third_corner[0],
                                                           self._previous_third_corner[1])
                    distances_third.append(dict(index=c, point=(x, y), distance=distance_third))

        distances_default = sorted(distances_default, key=lambda item: item['distance'])

        # search for second-corner
        second_corner, distances_default = self._trackCorner(distances_second, distances_default)
        self._previous_second_corner = second_corner

        # search for third-corner
        third_corner, distances_default = self._trackCorner(distances_third, distances_default)
        self._previous_third_corner = third_corner

        fourth_corner = None
        # search for fourth-corner
        for corner in corners:
            x, y = corner.ravel()
            if (x != default_corner[0] or y != default_corner[1]) \
                    and (x != second_corner[0] or y != second_corner[1]) \
                    and (x != third_corner[0] or y != third_corner[1]):
                fourth_corner = (x, y)
                break

        return second_corner, third_corner, fourth_corner

    @staticmethod
    def _findDefaultCorner(img, corners):
        """
        Find the default corner of the rectangle
        The default corner is the nearest to the circle into the shape
        :param img: OpenCv image
        :param corners: corners of the rectangle
        :return:
        """

        # search default point
        show_img = None
        # show_img = img.copy() # debug
        default_corner = None
        min_distance = 100
        height, width, _ = img.shape
        x_median, y_median = FeatureMatcher._getCornersCenter(corners, height, width)
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

        return default_corner

    @staticmethod
    def _getCornersCenter(corners, height, width):
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
    def _trackCorner(distances_previous_point, distances_default):
        """
        Track the corner point nearest to the previous
        :param distances_previous_point: distances from the previous point and the given corners
        :param distances_default: distances to the default corner, ordered by distance
        :return:
        """

        if len(distances_previous_point) <= 0:
            # set current corner to the nearest to the default
            current_corner = distances_default[0].get('point')
            # remove current corner distance to avoid re-picking
            distances_default.pop(0)
        else:
            # select the one nearest to the previous point as the current corner
            distances_previous_point = sorted(distances_previous_point, key=lambda item: item['distance'])
            current_corner = distances_previous_point[0].get('point')

        return current_corner, distances_default

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
                    cv2.putText(test_img, "perimeter: {}".format(peri), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                (255, 255, 255), 3, cv2.LINE_AA)
                    cv2.imshow("test{}".format(i), cv2.resize(test_img, None, fx=0.6, fy=0.6))
                if peri > max_perimeter:
                    max_perimeter = peri
                    cnt = [c]
                cnts_approx.append(c)
                perimeters.append(peri)

        if max_only:
            if cnt is None:
                return None
            else:
                return np.array(cnt)

        return np.array(cnts_approx)

    @staticmethod
    def _getStaticPixelsIntensity(static_img, static_shape_points, area_diameter=300):
        h, w, _ = static_img.shape
        corners = np.array(static_shape_points)
        x_center, y_center = FeatureMatcher._getCornersCenter(corners, h, w)
        x_min, x_max = x_center - area_diameter, x_center + area_diameter
        y_min, y_max = y_center - area_diameter, y_center + area_diameter

        if x_min < 0:
            x_min = 1
        if x_max > w:
            x_max = w - 1
        if y_min < 0:
            y_min = 1
        if y_max > h:
            y_max = h - 1

        roi_img = static_img[y_min:y_max, x_min:x_max]

        return roi_img
