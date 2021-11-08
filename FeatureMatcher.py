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
        """
        Constructor
        :param frames_static_folder_path: path to the static frames folder
        :param frames_moving_folder_path: path to the moving frames folder
        :param detector_algorithm: detector algorithm to use
        :param matching_algorithm: matching algorithm to use
        :param algorithm_params: algorithm params
        """
        self.frames_static_folder_path = frames_static_folder_path
        self.frames_moving_folder_path = frames_moving_folder_path
        self.detector_algorithm = detector_algorithm
        self.matching_algorithm = matching_algorithm
        if algorithm_params is not None:
            self.algorithm_params = algorithm_params
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    def setOrbThreshold(self, matcher):
        """
        Set default threshold for ORB
        :param matcher:
        """
        if matcher == self.MATCHING_ALGORITHM_KNN:
            self.algorithm_params = dict(min_match=40, threshold=0.85)
        elif matcher == self.MATCHING_ALGORITHM_FLANN:
            self.algorithm_params = dict(min_match=20, threshold=0.85)
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    def setSiftThreshold(self, matcher):
        """
        Set default threshold for SIFT
        :param matcher:
        """
        if matcher == self.MATCHING_ALGORITHM_FLANN:
            self.algorithm_params = dict(min_match=15, threshold=0.8)
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    @staticmethod
    def homography_check(train_image, homography_image):
        """
        Check if the homography image match with the train one
        :param train_image: OpenCv image
        :param homography_image: OpenCv image
        :return:
        """
        detector_alg = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        train_img_bw = cv2.cvtColor(train_image, cv2.COLOR_BGR2GRAY)
        query_img_bw = cv2.cvtColor(homography_image, cv2.COLOR_BGR2GRAY)

        queryKeypoints, queryDescriptors = detector_alg.detectAndCompute(query_img_bw, None)
        trainKeypoints, trainDescriptors = detector_alg.detectAndCompute(train_img_bw, None)
        matches = matcher.knnMatch(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors, k=2)
        # Apply Lowe ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)

        if len(good_matches) < 10:
            return False
        return True

    @staticmethod
    def homography_transformation(query_image, query_features, train_image, train_features, matches,
                                  transform_train=True, show_images=True, save_as=None):
        """
        Find homography matrix and do perspective transform
        :param query_image: OpenCv image
        :param query_features: keypoints, descriptors of the query_image
        :param train_image: OpenCv image
        :param train_features: keypoints, descriptors of the train_image
        :param matches: matches between train_image and query_image
        :param transform_train: if True transform train into query image, otherwise transform query into train image
        :param show_images: if True show results
        :param save_as: save filename for results, if None don't save
        :return:
        """
        import os

        kp_query_image, desc_query_image = query_features[0], query_features[1]
        kp_train_image, desc_train_image = train_features[0], train_features[1]

        query_pts = np.float32([kp_query_image[m.queryIdx]
                               .pt for m in matches]).reshape(-1, 1, 2)
        train_pts = np.float32([kp_train_image[m.trainIdx]
                               .pt for m in matches]).reshape(-1, 1, 2)

        if transform_train:
            # Warp train image into query image based on homography
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 5.0)
            im_out = cv2.warpPerspective(train_image, matrix, (query_image.shape[1], query_image.shape[0]))
        else:
            # Warp train image into query image based on homography
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 5.0)
            im_out = cv2.warpPerspective(query_image, matrix, (train_image.shape[1], train_image.shape[0]))

        if show_images:
            cv2.imshow("Transformed", im_out)
            # cv2.waitKey(0)
        if save_as is not None:
            if not os.path.isdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH):
                os.mkdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH)
            cv2.imwrite(cst.TRANSFORMATION_RESULTS_FOLDER_PATH + '/' + save_as, im_out)

        if FeatureMatcher.homography_check(train_image, im_out):
            return matrix
        else:
            return None

    def prepareMatcher(self):
        """
        Prepare matcher and detector algorithms
        :return:
        """
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

    def extract_features(self, show_images=False, save_images=False, plot_histogram=False):
        """
        Feature matching and homography check
        :param show_images: if True show results
        :param save_images: if True save results
        :param plot_histogram: if True plot light intensity
        :return:
        """
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
            train_filename = self.frames_static_folder_path + "/frame_{}.png".format(i)
            train_img = cv2.imread(train_filename)
            train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
            # Read the query image
            # The query image is what we need to find in train image
            query_filename = self.frames_moving_folder_path + "/frame_{}.png".format(i)
            query_img = cv2.imread(query_filename)
            query_img = ut.enchant_brightness_and_contrast(query_img)
            query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

            train_img_bw = ut.enchant_morphological(train_img_bw, [cv2.MORPH_OPEN], iterations=5)
            query_img_bw = ut.enchant_morphological(query_img_bw, [cv2.MORPH_OPEN])

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

                homography = FeatureMatcher.homography_transformation(query_image=query_img,
                                                                      query_features=(queryKeypoints, queryDescriptors),
                                                                      train_image=train_img,
                                                                      train_features=(trainKeypoints, trainDescriptors),
                                                                      matches=good_matches, show_images=show_images,
                                                                      save_as=save_as)
                if homography is not None:
                    n_accepted += 1
                    #print(ut.cameraPoseFromHomography(homography))
                    data = dict(trainImage=train_img,
                                queryImage=query_img,
                                trainFeatures=(trainKeypoints, trainDescriptors),
                                queryFeatures=(queryKeypoints, queryDescriptors),
                                homography=homography)
                    dataset.append(data)
                else:
                    n_discarded += 1
                    print("Discarded: Inaccurate homography")

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
                print("Discarded: Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH))

        print("\nN° of accepted frames: ", n_accepted)
        print("N° of discarded frames: ", n_discarded, "\n")

        return dataset
