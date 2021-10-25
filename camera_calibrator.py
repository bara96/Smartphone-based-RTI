# Import required modules
import os
import random
import shutil
import cv2
import numpy as np


# generate the frames images from the video, return the n° of frames generated
# video_path: video path
# frames_dir: path where to save frames
# from_scratch: if true will regenerate the entire frame sequence from scratch
def generate_video_frames(video_path, frames_dir='assets/frames', from_scratch=True):
    # delete previous saved frames images
    if from_scratch:
        if os.path.exists(frames_dir):
            try:
                shutil.rmtree(frames_dir)
                os.mkdir(frames_dir)
            except OSError as e:
                print("Error: %s - %s." % (e.filename, e.strerror))
        else:
            os.mkdir(frames_dir)

    # Opens the Video file
    cap = cv2.VideoCapture(video_path)
    i = n_generated = n_skipped = 0
    print('Generating frames.. \n')
    while (cap.isOpened()):
        skip = False
        ret, frame = cap.read()
        if ret == False:
            break
        frame_path = frames_dir + '/frame_' + str(i) + '.png'
        # skip already generated frames
        if not from_scratch:
            if os.path.exists(frame_path):
                skip = True
        if not skip:
            cv2.imwrite(frame_path, frame)
            n_generated += 1
        else:
            n_skipped += 1
        i += 1
    print('Generated {:d} frames.. \n'.format(n_generated))
    if not from_scratch:
        print('Skipped {:d} frames.. \n'.format(n_skipped))

    cap.release()
    cv2.destroyAllWindows()
    return i


# Calibrate the camera reading frames_dir from generated assets
# n_frames: n° of random frames_dir to use for the calibration
# frames_dir: path where to get frames
# show_images: if True, display the calibrated images
def calibrate_camera(frames_dir='assets/frames/', n_frames=20, show_images=True):
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

    gray_color = None

    #  3D points real world coordinates
    objectp3d = np.zeros((1, CHECKERBOARD[0]
                          * CHECKERBOARD[1],
                          3), np.float32)
    objectp3d[0, :, :2] = np.mgrid[0:CHECKERBOARD[0],
                          0:CHECKERBOARD[1]].T.reshape(-1, 2)
    prev_img_shape = None

    # Extracting path of random images stored in a given directory.
    # images = glob.glob('%s/frame*.png' % frames_dir)
    for i in range(n_frames):
        filename = random.choice(os.listdir(frames_dir))
        print("Selected: ", frames_dir + filename)
        image = cv2.imread(frames_dir + filename)
        gray_color = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Find the chess board corners
        # If desired number of corners are
        # found in the image then ret = true
        ret, corners = cv2.findChessboardCorners(
            gray_color, CHECKERBOARD,
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
                gray_color, corners, (11, 11), (-1, -1), criteria)

            twodpoints.append(corners2)

            # Draw and display the corners
            image = cv2.drawChessboardCorners(image, CHECKERBOARD, corners2, ret)

        if show_images:
            cv2.imshow('img', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

    # Perform camera calibration by
    # passing the value of above found out 3D points (threedpoints)
    # and its corresponding pixel coordinates of the
    # detected corners (twodpoints)
    print("\n\n")
    print("Calibrating...")
    (ret, matrix, distortion, r_vecs, t_vecs) = cv2.calibrateCamera(threedpoints, twodpoints, gray_color.shape[::-1],
                                                                    None, None)

    # Displaying required output
    print("\n\nCamera matrix: \n")
    print(matrix)

    print("\n\nDistortion coefficient: \n")
    print(distortion)

    print("\n\nRotation Vectors: \n")
    print(r_vecs)

    print("\n\nTranslation Vectors: \n")
    print(t_vecs)

    # Write instrinsics to file
    Kfile = cv2.FileStorage('assets/intrinsics.xml', cv2.FILE_STORAGE_WRITE)
    Kfile.write("K", matrix)
    Kfile.write("distortion", distortion)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dir1 = 'assets/G3DCV2021_data/cam1 - static/calibration.mov'
    dir2 = 'assets/G3DCV2021_data/cam2 - moving light/calibration.mp4'
    # generate_video_frames(dir1, 'assets/frames_calibration')
    calibrate_camera('assets/frames_calibration/')
