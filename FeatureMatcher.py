import os
import cv2
import constants as cst
from matplotlib import pyplot as plt
import utilities as ut
import numpy as np


class FeatureMatcher:
    DETECTOR_ALGORITHM_ORB = 'orb'
    DETECTOR_ALGORITHM_SIFT = 'sift'

    MATCHING_ALGORITHM_BRUTEFORCE = 'bruteforce'
    MATCHING_ALGORITHM_KNN = 'knn'
    MATCHING_ALGORITHM_FLANN = 'flann'

    def __init__(self, frames_static_folder_path, frames_moving_folder_path,
                 detector_algorithm=DETECTOR_ALGORITHM_ORB,
                 matching_algorithm=MATCHING_ALGORITHM_BRUTEFORCE,
                 algorithm_params=None):
        self.frames_static_folder_path = frames_static_folder_path
        self.frames_moving_folder_path = frames_moving_folder_path
        self.detector_algorithm = detector_algorithm
        self.matching_algorithm = matching_algorithm
        self.algorithm_params = algorithm_params

    def setOrbTreshold(self, matcher):
        if matcher == self.MATCHING_ALGORITHM_KNN:
            self.algorithm_params = dict(min_match=40, threshold=0.85)
        elif matcher == self.MATCHING_ALGORITHM_FLANN:
            self.algorithm_params = dict(min_match=20, threshold=0.85)
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    def setSiftTreshold(self, matcher):
        if matcher == self.MATCHING_ALGORITHM_FLANN:
            self.algorithm_params = dict(min_match=15, threshold=0.8)
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    # prepare matcher
    def prepareMatcher(self):
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH = 6

        if self.detector_algorithm == self.DETECTOR_ALGORITHM_ORB:
            # Initialize the ORB detector algorithm
            detector_alg = cv2.ORB_create()
            if self.matching_algorithm == self.MATCHING_ALGORITHM_BRUTEFORCE:
                # feature matching using Brute-Force matching with ORB Descriptors
                # Initialize the Matcher for matching the keypoints
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            elif self.matching_algorithm == self.MATCHING_ALGORITHM_KNN:
                # feature matching using KNN matching with ORB Descriptors
                # Initialize the Matcher for matching the keypoints
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            elif self.matching_algorithm == self.MATCHING_ALGORITHM_FLANN:
                # feature matching using FLANN matching with ORB Descriptors
                index_params = dict(algorithm=FLANN_INDEX_LSH,
                                    table_number=6,  # 12
                                    key_size=12,  # 20
                                    multi_probe_level=2)  # 2
                # It specifies the number of times the trees in the index should be recursively traversed.
                # Higher values gives better precision, but also takes more time
                search_params = dict(checks=60)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise Exception('Matching algorithm for orb not recognised!')
        elif self.detector_algorithm == self.DETECTOR_ALGORITHM_SIFT:
            # Initialize the SIFT detector algorithm
            detector_alg = cv2.SIFT_create()
            if self.matching_algorithm == self.MATCHING_ALGORITHM_FLANN:
                # feature matching using FLANN matching with SIFT Descriptors
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                # It specifies the number of times the trees in the index should be recursively traversed.
                # Higher values gives better precision, but also takes more time
                search_params = dict(checks=60)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise Exception('Matching algorithm for sift not recognised!')
        else:
            raise Exception('Detector algorithm not recognised!')

        return detector_alg, matcher

    # feature matching and homography transformations
    def extract_features(self, show_images=False, save_images=False, plot_histogram=False):
        if not os.path.isdir(self.frames_static_folder_path):
            raise Exception('Static folder not found!')
        if not os.path.isdir(self.frames_moving_folder_path):
            raise Exception('Moving folder not found!')

        # read frames from folders
        list_static = os.listdir(self.frames_static_folder_path)
        n_files_static = len(list_static)
        list_moving = os.listdir(self.frames_moving_folder_path)
        n_files_moving = len(list_moving)
        tot_frames = min(n_files_static, n_files_moving)

        detector_alg, matcher = self.prepareMatcher()

        # default algorithm_params
        MIN_MATCH = 10
        THRESHOLD = 0.75
        if self.algorithm_params['min_match'] is not None:
            MIN_MATCH = self.algorithm_params['min_match']
        if MIN_MATCH < 4:
            MIN_MATCH = 4  # required at least 4 matches for homography
        if self.algorithm_params['threshold'] is not None:
            THRESHOLD = self.algorithm_params['threshold']

        print("Selected parameters:")
        print("- Detector Algorithm: ", self.detector_algorithm)
        print("- Matching Algorithm: ", self.matching_algorithm)
        print("- MIN MATCH: ", MIN_MATCH)
        print("- THRESHOLD: ", THRESHOLD)
        print("\n")

        dataset = []
        n_accepted = 0
        n_discarded = 0
        for i in range(0, tot_frames):
            # Read the train image
            train_img = cv2.imread(self.frames_static_folder_path + "/frame_{}.png".format(i))
            train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
            # Read the query image
            # The query image is what we need to find in train image
            query_img = cv2.imread(self.frames_moving_folder_path + "/frame_{}.png".format(i))
            query_img = ut.enchant_brightness_and_contrast(query_img)
            query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

            train_img_bw = ut.image_enchantment(train_img_bw, [cv2.MORPH_OPEN], 5)
            query_img_bw = ut.image_enchantment(query_img_bw, [cv2.MORPH_OPEN])

            # Now detect the keypoints and compute the descriptors for the query image and train image
            queryKeypoints, queryDescriptors = detector_alg.detectAndCompute(query_img_bw, None)
            trainKeypoints, trainDescriptors = detector_alg.detectAndCompute(train_img_bw, None)

            if plot_histogram:
                histr = cv2.calcHist([train_img], [0], None, [256], [0, 256])
                plt.plot(histr)
                plt.show(block=False)

                histr = cv2.calcHist([query_img], [0], None, [256], [0, 256])
                plt.plot(histr)
                plt.show(block=False)

            # match the keypoints and sort them in the order of their distance.
            if self.detector_algorithm == self.DETECTOR_ALGORITHM_ORB and self.matching_algorithm == self.MATCHING_ALGORITHM_BRUTEFORCE:
                matches = matcher.match(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors)
                good_matches = matches
            else:
                matches = matcher.knnMatch(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors, k=2)
                # Apply Lowe ratio test
                good_matches = []
                for m, n in matches:
                    if m.distance < THRESHOLD * n.distance:
                        good_matches.append(m)

            good_matches = sorted(good_matches, key=lambda x: x.distance)

            if len(good_matches) >= MIN_MATCH:
                print("Matches found - %d/%d" % (len(good_matches), MIN_MATCH))
                # try to transform the static into the moving
                save_as = None
                if save_images:
                    save_as = "frame_{}.png".format(i)

                homography = ut.homography_transformation(query_image=query_img,
                                                          query_features=(queryKeypoints, queryDescriptors),
                                                          train_image=train_img,
                                                          train_features=(trainKeypoints, trainDescriptors),
                                                          matches=good_matches, show_images=show_images,
                                                          save_as=save_as)
                if homography is not None:
                    n_accepted += 1
                    dataset.append((queryKeypoints, queryDescriptors, homography))
                else:
                    n_discarded += 1
                    print("Inaccurate homography")

                # draw the matches to the final image containing both the images
                final_img = cv2.drawMatches(query_img_bw, queryKeypoints, train_img_bw, trainKeypoints, good_matches,
                                            None)
                final_img = cv2.resize(final_img, (1000, 650))

                # Show the final image
                if show_images:
                    cv2.imshow("Matches", final_img)
                    cv2.waitKey(0)
                    plt.close()
                # Save the final image
                if save_images:
                    if not os.path.isdir(cst.MATCHING_RESULTS_FOLDER_PATH):
                        os.mkdir(cst.MATCHING_RESULTS_FOLDER_PATH)
                    cv2.imwrite(cst.MATCHING_RESULTS_FOLDER_PATH + '/frame_{}.png'.format(i), final_img)
            else:
                n_discarded += 1
                print("Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH))

        print("\nN° of accepted frames: ", n_accepted)
        print("N° of discarded frames: ", n_discarded, "\n")
