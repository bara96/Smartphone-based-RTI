# Import required modules
import constants as cst
import os
import cv2
import numpy as np
from timeit import default_timer as timer


def calibrate(video_path, save_path, frame_skip=60, show_images=True):
    """
    Calibrate the camera reading video frames
    :param video_path: path to the video
    :param save_path: path where to save the intrinsics
    :param frame_skip: set how many frame to skip between each calibration
    :param show_images: if True show the calibrated images
    """
    if not os.path.isfile(video_path):
        raise Exception('Video not found!')

    # Define the dimensions of checkerboard
    CHECKERBOARD = (6, 9)

    # stop the iteration when specified
    # accuracy, epsilon, is reached or
    # specified number of iterations are completed.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Vector for 3D points
    threedpoints = []

    # Vector for 2D points
    twodpoints = []

    image_gray = None

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                          * CHECKERBOARD[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)

    print("Calibrating... \n")
    n_frames_read = 0
    n_frame = 0
    video = cv2.VideoCapture(video_path)
    while video.isOpened():
        ret, image = video.read()
        n_frame += frame_skip
        video.set(1, n_frame)  # grab a frame every n
        if ret:
            n_frames_read += 1
            image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            # If desired number of corners are
            # found in the image then ret = true
            ret, corners = cv2.findChessboardCorners(
                image_gray, CHECKERBOARD,
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE)

            # If desired number of corners can be detected then,
            # refine the pixel coordinates and display
            # them on the images of checker board
            if ret == True:
                threedpoints.append(objectp3d)

                # Refining pixel coordinates
                # for given 2d points.
                corners2 = cv2.cornerSubPix(
                    image_gray, corners, (11, 11), (-1, -1), criteria)

                twodpoints.append(corners2)

                # Draw and display the corners
                image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

            if show_images:
                cv2.imshow('img', image)
                cv2.waitKey(0)
        else:
            video.release()
            break
    video.release()
    cv2.destroyAllWindows()

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)

    (ret, matrix, distortion, r_vecs, t_vecs) = cv2.calibrateCamera(threedpoints, twodpoints, image_gray.shape[::-1],
                                                                    None, None)

    print("Calibration ended... \n")
    # Displaying required output
    print("Camera matrix: \n")
    print(matrix)

    print("\n\nDistortion coefficient: \n")
    print(distortion)

    print("\n\nRotation Vectors: \n")
    print(r_vecs)

    print("\n\nTranslation Vectors: \n")
    print(t_vecs)

    # Write intrinsics to file
    Kfile = cv2.FileStorage(save_path, cv2.FILE_STORAGE_WRITE)
    Kfile.write("K", matrix)
    Kfile.write("distortion", distortion)
    print("Calibration saved: \"{}\" \n".format(save_path))


def compute():
    """
    Main function
    """
    calibrate(cst.ASSETS_STATIC_FOLDER + '/calibration.mov', save_path=cst.INTRINSICS_STATIC_PATH, show_images=False)
    calibrate(cst.ASSETS_MOVING_FOLDER + '/calibration.mp4', save_path=cst.INTRINSICS_MOVING_PATH, show_images=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    start = timer()
    compute()
    print("Computation duration: {} s".format(round(timer() - start, 2)))
