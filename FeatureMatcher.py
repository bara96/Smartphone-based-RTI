import os
import cv2
import constants as cst
from matplotlib import pyplot as plt
import utilities as ut
import numpy as np


class FeatureMatcher:
    DETECTOR_ALGORITHM_ORB = 'ORB'
    DETECTOR_ALGORITHM_SIFT = 'SIFT'
    DETECTOR_ALGORITHM_SURF = 'SUFT'

    MATCHING_ALGORITHM_BRUTEFORCE = 'BRUTEFORCE'
    MATCHING_ALGORITHM_KNN = 'KNN'
    MATCHING_ALGORITHM_FLANN = 'FLANN'

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

    def setThreshold(self, matcher):
        """
        Set a default pre-saved threshold for Lowe ratio test
        :param matcher: matcher algorithm
        """
        if matcher == FeatureMatcher.MATCHING_ALGORITHM_KNN:
            self.algorithm_params = dict(min_match=10, threshold=0.75)
        elif matcher == FeatureMatcher.MATCHING_ALGORITHM_FLANN:
            self.algorithm_params = dict(min_match=10, threshold=0.80)
        else:
            self.algorithm_params = dict(min_match=10, threshold=0.75)

    def prepareMatcher(self):
        """
        Prepare matcher and detector algorithms
        :return:
        """
        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        FLANN_INDEX_LSH = 6

        if self.detector_algorithm == FeatureMatcher.DETECTOR_ALGORITHM_ORB:
            # Initialize the ORB detector algorithm and the Matcher for matching the keypoints
            detector_alg = cv2.ORB_create()
            if self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_BRUTEFORCE:
                # feature matching using Brute-Force matching with ORB Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_KNN:
                # feature matching using KNN matching with ORB Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_FLANN:
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
                raise Exception('Matching algorithm for ORB not recognised!')
        elif self.detector_algorithm == FeatureMatcher.DETECTOR_ALGORITHM_SIFT:
            # Initialize the SIFT detector algorithm and the Matcher for matching the keypoints
            detector_alg = cv2.SIFT_create()
            if self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_BRUTEFORCE:
                # feature matching using Brute-Force matching with SIFT Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_KNN:
                # feature matching using KNN matching with SIFT Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_FLANN:
                # feature matching using FLANN matching with SIFT Descriptors
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                # It specifies the number of times the trees in the index should be recursively traversed.
                # Higher values gives better precision, but also takes more time
                search_params = dict(checks=60)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise Exception('Matching algorithm for SIFT not recognised!')
        elif self.detector_algorithm == FeatureMatcher.DETECTOR_ALGORITHM_SURF:
            # Initialize the SURF detector algorithm and the Matcher for matching the keypoints
            detector_alg = cv2.xfeatures2d.SURF_create()
            if self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_BRUTEFORCE:
                # feature matching using Brute-Force matching with SURF Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_KNN:
                # feature matching using KNN matching with SURF Descriptors
                matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            elif self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_FLANN:
                # feature matching using FLANN matching with SURF Descriptors
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                # It specifies the number of times the trees in the index should be recursively traversed.
                # Higher values gives better precision, but also takes more time
                search_params = dict(checks=60)
                matcher = cv2.FlannBasedMatcher(index_params, search_params)
            else:
                raise Exception('Matching algorithm for SURF not recognised!')
        else:
            raise Exception('Detector algorithm not recognised!')

        return detector_alg, matcher

    @staticmethod
    def _homographyCheck(train_image, homography_image):
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

        train_img_bw = cv2.GaussianBlur(train_img_bw, (5, 5), 0)
        query_img_bw = cv2.GaussianBlur(query_img_bw, (5, 5), 0)

        queryKeypoints, queryDescriptors = detector_alg.detectAndCompute(query_img_bw, None)
        trainKeypoints, trainDescriptors = detector_alg.detectAndCompute(train_img_bw, None)
        matches = matcher.knnMatch(queryDescriptors=queryDescriptors, trainDescriptors=trainDescriptors, k=2)

        # Apply Lowe ratio test
        good_matches = []
        for m, n in matches:
            if m.distance < 0.70 * n.distance:
                good_matches.append(m)
        good_matches = sorted(good_matches, key=lambda x: x.distance)

        print("Checks:", len(good_matches))
        if len(good_matches) < 25:
            return False
        return True

    @staticmethod
    def _homographyTransformation(query_image, query_features, train_image, train_features, matches,
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
            matrix, mask = cv2.findHomography(train_pts, query_pts, cv2.RANSAC, 8)
            if matrix is None:
                return None
            im_out = cv2.warpPerspective(train_image, matrix, (query_image.shape[1], query_image.shape[0]))
        else:
            # Warp train image into query image based on homography
            matrix, mask = cv2.findHomography(query_pts, train_pts, cv2.RANSAC, 8)
            if matrix is None:
                return None
            im_out = cv2.warpPerspective(query_image, matrix, (train_image.shape[1], train_image.shape[0]))

        if show_images:
            cv2.imshow("Transformed", cv2.resize(im_out, None, fx=0.3, fy=0.3))
            #cv2.waitKey(0)
        if save_as is not None:
            if not os.path.isdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH):
                os.mkdir(cst.TRANSFORMATION_RESULTS_FOLDER_PATH)
            cv2.imwrite(cst.TRANSFORMATION_RESULTS_FOLDER_PATH + '/' + save_as, im_out)

        if FeatureMatcher._homographyCheck(im_out, train_image):
            return matrix
        else:
            return None

    @staticmethod
    def _checkOutliers(train_keypoints, query_keypoints, matches):
        """
        Usage of Ransac to remove outliers from matches
        :param train_keypoints: keypoints of the train image
        :param query_keypoints: keypoints of the query image
        :param matches: matches between the keypoints of  the two images
        :return:
        """
        from skimage.measure import ransac
        from skimage.transform import AffineTransform

        MIN_SAMPLES = 4
        if len(matches) <= MIN_SAMPLES:
            return train_keypoints, query_keypoints, matches, False

        train_pts = np.float32([train_keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 2)
        query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 2)
        # Ransac
        model, inliers = ransac(
            (train_pts, query_pts),
            AffineTransform, min_samples=MIN_SAMPLES,
            residual_threshold=8, max_trials=500
        )

        n_inliers = np.sum(inliers)

        inlier_keypoints_train = [cv2.KeyPoint(point[0], point[1], 1) for point in train_pts[inliers]]
        inlier_keypoints_query = [cv2.KeyPoint(point[0], point[1], 1) for point in query_pts[inliers]]
        placeholder_matches = [cv2.DMatch(idx, idx, 1) for idx in range(n_inliers)]

        return inlier_keypoints_train, inlier_keypoints_query, placeholder_matches, True

    @staticmethod
    def _findPosePNP(train_keypoints, query_keypoints, matches, K, d, image, show_position=True):
        """
        Solve PNP and use Rodrigues to find Camera world position
        :param train_keypoints: keypoints of the train image
        :param query_keypoints: keypoints of the query image
        :param matches: matches between the keypoints of  the two images
        :param K: camera intrinsics matrix
        :param d: camera intrinsics distortion
        :param image: OpenCv image
        :param show_position:
        :return:
        """
        # train_pts = np.float32([train_keypoints[m.trainIdx].pt for m in matches])
        train_pts = np.float32([np.append(train_keypoints[m.trainIdx].pt, 1.) for m in matches])
        query_pts = np.float32([query_keypoints[m.queryIdx].pt for m in matches])
        # query_pts = np.float32([np.append(query_keypoints[m.queryIdx].pt, 1.) for m in matches])

        ret, rvecs, tvecs = cv2.solvePnP(train_pts, query_pts, K, d)
        rotM = cv2.Rodrigues(rvecs)[0]
        camera_position = -np.matrix(rotM).T * np.matrix(tvecs)
        #camera_position = -rotM.transpose() * tvecs

        train_img_new = image.copy()
        train_img_new, x, y = ut.image_draw_circle(train_img_new, camera_position[0], camera_position[1],
                                                   (0, 0, 255))
        if show_position:
            cv2.imshow("Camera Position", cv2.resize(train_img_new, None, fx=0.3, fy=0.3))

        return camera_position

    def extractFeatures(self, show_params=None, save_images=False):
        """
        Feature matching and homography check
        :param show_params: if True show all results
        :param save_images: if True save results
        :return:
        """

        if not os.path.isdir(self.frames_static_folder_path):
            raise Exception('Static folder not found!')
        if not os.path.isdir(self.frames_moving_folder_path):
            raise Exception('Moving folder not found!')

        show_homography = False
        show_camera_position = False
        show_matches = False
        show_histogram = False
        if show_params is True:
            show_homography = True
            show_camera_position = True
            show_matches = True
            show_histogram = True
        elif show_params is not None:
            if show_params['homography'] is not None:
                show_homography = show_params['homography']
            if show_params['camera_position'] is not None:
                show_camera_position = show_params['camera_position']
            if show_params['matches'] is not None:
                show_matches = show_params['matches']
            if show_params['histogram'] is not None:
                show_histogram = show_params['histogram']

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

            # Read the query image
            # The query image is what we need to find in train image
            query_filename = self.frames_moving_folder_path + "/frame_{}.png".format(i)
            query_img = cv2.imread(query_filename)

            # calculate brightness histogram
            if show_histogram:
                histr = cv2.calcHist([train_img], [0], None, [256], [0, 256])
                plt.plot(histr)
                plt.show(block=False)

                histr = cv2.calcHist([query_img], [0], None, [256], [0, 256])
                plt.plot(histr)
                plt.show(block=False)

            '''
            Image enchantments Phase
            '''
            train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)
            # query_img = ut.enchant_brightness_and_contrast(query_img)
            query_img_bw = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)

            #train_img_bw = ut.enchant_morphological(train_img_bw, [cv2.MORPH_OPEN])
            #query_img_bw = ut.enchant_morphological(query_img_bw, [cv2.MORPH_OPEN])

            for k in range(0, 5):
                train_img_bw = cv2.GaussianBlur(train_img_bw, (5, 5), 0)
                query_img_bw = cv2.GaussianBlur(query_img_bw, (5, 5), 0)

            '''
            Feature Matching Phase
            '''
            # Now detect the keypoints and compute the descriptors for the query image and train image
            queryKeypoints, queryDescriptors = detector_alg.detectAndCompute(query_img_bw, None)
            trainKeypoints, trainDescriptors = detector_alg.detectAndCompute(train_img_bw, None)

            # match the keypoints and sort them in the order of their distance.
            if self.matching_algorithm == FeatureMatcher.MATCHING_ALGORITHM_BRUTEFORCE:
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

            # remove outliers
            trainKeypoints, queryKeypoints, good_matches, check = FeatureMatcher._checkOutliers(train_keypoints=trainKeypoints,
                                                                                                query_keypoints=queryKeypoints,
                                                                                                matches=good_matches)
            ''' 
            Homography checks Phase 
            '''
            if not check:
                print("Discarded: Can't remove outliers")
            elif len(good_matches) >= MIN_MATCH:
                # try to transform the static into the moving
                save_as = None
                if save_images:
                    save_as = "frame_{}.png".format(i)

                # find homography to check if matches are acceptable
                homography = FeatureMatcher._homographyTransformation(query_image=query_img,
                                                                      query_features=(queryKeypoints, queryDescriptors),
                                                                      train_image=train_img,
                                                                      train_features=(trainKeypoints, trainDescriptors),
                                                                      matches=good_matches,
                                                                      show_images=show_homography,
                                                                      save_as=save_as)
                if homography is not None:
                    print("Accepted: matches found - %d/%d" % (len(good_matches), MIN_MATCH))
                    n_accepted += 1

                    K, d = ut.get_camera_intrinsics(cst.INTRINSICS_STATIC_PATH)

                    camera_position = FeatureMatcher._findPosePNP(trainKeypoints, queryKeypoints, good_matches,
                                                                  K, d, train_img, show_camera_position)

                    data = dict(trainImage=train_filename,
                                queryImage=query_filename,
                                camera_position=camera_position)
                    dataset.append(data)
                else:
                    n_discarded += 1
                    print("Discarded: Inaccurate homography")
            else:
                n_discarded += 1
                print("Discarded: Not enough matches are found - %d/%d" % (len(good_matches), MIN_MATCH))

            '''
            Show results Phase
            '''
            # draw the matches to the final image containing both the images
            final_img = None
            if show_matches or save_images:
                final_img = cv2.drawMatches(query_img_bw, queryKeypoints, train_img_bw, trainKeypoints, good_matches,
                                            None)

            # Show the final image
            if show_matches:
                cv2.imshow("Matches", cv2.resize(final_img, None, fx=0.4, fy=0.4))
            # Save the final image
            if save_images:
                if not os.path.isdir(cst.MATCHING_RESULTS_FOLDER_PATH):
                    os.mkdir(cst.MATCHING_RESULTS_FOLDER_PATH)
                cv2.imwrite(cst.MATCHING_RESULTS_FOLDER_PATH + '/frame_{}.png'.format(i), final_img)

            if show_params is not None:
                cv2.waitKey(0)
                plt.close()

        print("\n")
        print("N° of accepted frames: ", n_accepted)
        print("N° of discarded frames: ", n_discarded)
        print("Error %: ", n_discarded / tot_frames * 100)
        print("\n")

        return dataset
