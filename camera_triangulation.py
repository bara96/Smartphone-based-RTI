# Import required modules
import constants as cst
import cv2
import numpy as np
from matplotlib import pyplot as plt
import utilities as ut


def stereo_calibrate(mtx_static, dist_static, mtx_moving, dist_moving, frames_static_folder_path, frames_moving_folder_path):
    import os

    # read frames from folders
    list_static = os.listdir(frames_static_folder_path)
    n_files_static = len(list_static)
    list_moving = os.listdir(frames_moving_folder_path)
    n_files_moving = len(list_moving)
    tot_frames = min(n_files_static, n_files_moving)

    print("Reading frames")
    c1_images = []
    c2_images = []
    for i in range(0, tot_frames):
        train_filename = frames_static_folder_path + "/frame_{}.png".format(i)
        _im = cv2.imread(train_filename)
        c1_images.append(_im)

        query_filename = frames_moving_folder_path + "/frame_{}.png".format(i)
        _im = cv2.imread(query_filename)
        c2_images.append(_im)
    print("Frames read")


    # change this if stereo calibration not good.
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.0001)

    CHECKERBOARD = (6, 9)
    world_scaling = 1.  # change this to the real world square size. Or not.

    # coordinates of squares in the checkerboard world space
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    objp = world_scaling * objp

    # frame dimensions. Frames should be the same size.
    width = c1_images[0].shape[1]
    height = c1_images[0].shape[0]

    # Pixel coordinates of checkerboards
    imgpoints_left = []  # 2d points in image plane.
    imgpoints_right = []

    # coordinates of the checkerboard in checkerboard world space.
    objpoints = []  # 3d point in real world space

    print("Computing stereo points")
    for frame1, frame2 in zip(c1_images, c2_images):
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        c_ret1, corners1 = cv2.findChessboardCorners(gray1, CHECKERBOARD, None)
        c_ret2, corners2 = cv2.findChessboardCorners(gray2, CHECKERBOARD, None)

        if c_ret1 == True and c_ret2 == True:
            corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
            corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

            cv2.drawChessboardCorners(frame1, CHECKERBOARD, corners1, c_ret1)
            #cv2.imshow('img', frame1)

            cv2.drawChessboardCorners(frame2, CHECKERBOARD, corners2, c_ret2)
            #cv2.imshow('img2', frame2)
            #cv2.waitKey(0)

            objpoints.append(objp)
            imgpoints_left.append(corners1)
            imgpoints_right.append(corners2)

    print("Stereo points computed")
    stereocalibration_flags = cv2.CALIB_FIX_INTRINSIC
    ret, CM1, dist_static, CM2, dist_moving, R, T, E, F = cv2.stereoCalibrate(objpoints, imgpoints_left, imgpoints_right, mtx_static,
                                                                              dist_static,
                                                                              mtx_moving, dist_moving, (width, height), criteria=criteria,
                                                                              flags=stereocalibration_flags)

    print(ret)
    return R, T


def triangulate(mtx1, mtx2, R, T):
    print("Computing triangulation")
    uvs1 = [[458, 86], [451, 164], [287, 181],
            [196, 383], [297, 444], [564, 194],
            [562, 375], [596, 520], [329, 620],
            [488, 622], [432, 52], [489, 56]]

    uvs2 = [[540, 311], [603, 359], [542, 378],
            [525, 507], [485, 542], [691, 352],
            [752, 488], [711, 605], [549, 651],
            [651, 663], [526, 293], [542, 290]]

    uvs1 = np.array(uvs1)
    uvs2 = np.array(uvs2)

    frame1 = cv2.imread(cst.FRAMES_FOLDER_PATH + '/static_coin1/frame_0.png')
    frame2 = cv2.imread(cst.FRAMES_FOLDER_PATH + '/moving_coin1/frame_0.png')

    plt.imshow(frame1[:, :, [2, 1, 0]])
    plt.scatter(uvs1[:, 0], uvs1[:, 1])
    plt.show()  # this call will cause a crash if you use cv2.imshow() above. Comment out cv2.imshow() to see this.

    plt.imshow(frame2[:, :, [2, 1, 0]])
    plt.scatter(uvs2[:, 0], uvs2[:, 1])
    plt.show()  # this call will cause a crash if you use cv2.imshow() above. Comment out cv2.imshow() to see this

    # RT matrix for C1 is identity.
    RT1 = np.concatenate([np.eye(3), [[0], [0], [0]]], axis=-1)
    P1 = mtx1 @ RT1  # projection matrix for C1

    # RT matrix for C2 is the R and T obtained from stereo calibration.
    RT2 = np.concatenate([R, T], axis=-1)
    P2 = mtx2 @ RT2  # projection matrix for C2

    def DLT(P1, P2, point1, point2):

        A = [point1[1] * P1[2, :] - P1[1, :],
             P1[0, :] - point1[0] * P1[2, :],
             point2[1] * P2[2, :] - P2[1, :],
             P2[0, :] - point2[0] * P2[2, :]
             ]
        A = np.array(A).reshape((4, 4))
        # print('A: ')
        # print(A)

        B = A.transpose() @ A
        from scipy import linalg
        U, s, Vh = linalg.svd(B, full_matrices=False)

        print('Triangulated point: ')
        print(Vh[3, 0:3] / Vh[3, 3])
        return Vh[3, 0:3] / Vh[3, 3]

    p3ds = []
    for uv1, uv2 in zip(uvs1, uvs2):
        _p3d = DLT(P1, P2, uv1, uv2)
        p3ds.append(_p3d)
    p3ds = np.array(p3ds)

    from mpl_toolkits.mplot3d import Axes3D

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim3d(-15, 5)
    ax.set_ylim3d(-10, 10)
    ax.set_zlim3d(10, 30)

    connections = [[0, 1], [1, 2], [2, 3], [3, 4], [1, 5], [5, 6], [6, 7], [1, 8], [1, 9], [2, 8], [5, 9], [8, 9],
                   [0, 10], [0, 11]]
    for _c in connections:
        print(p3ds[_c[0]])
        print(p3ds[_c[1]])
        ax.plot(xs=[p3ds[_c[0], 0], p3ds[_c[1], 0]], ys=[p3ds[_c[0], 1], p3ds[_c[1], 1]],
                zs=[p3ds[_c[0], 2], p3ds[_c[1], 2]], c='red')
    ax.set_title('This figure can be rotated.')
    # uncomment to see the triangulated pose. This may cause a crash if youre also using cv2.imshow() above.
    plt.show()


def compute(video_name='calibration'):
    import analysis
    video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(video_name)
    video_moving_path = cst.ASSETS_MOVING_FOLDER + '/{}.mp4'.format(video_name)
    frames_static_folder = cst.FRAMES_FOLDER_PATH + '/static_{}'.format(video_name)
    frames_moving_folder = cst.FRAMES_FOLDER_PATH + '/moving_{}'.format(video_name)

    mtx1, dist1 = ut.get_camera_intrinsics(cst.INTRINSICS_STATIC_PATH)
    mtx2, dist2 = ut.get_camera_intrinsics(cst.INTRINSICS_MOVING_PATH)

    analysis.sync_videos(video_static_path, video_moving_path, n_frames=30)

    R, T = stereo_calibrate(mtx1, dist1, mtx2, dist2, frames_static_folder_path=frames_static_folder, frames_moving_folder_path=frames_moving_folder)


    # this call might cause segmentation fault error. This is due to calling cv2.imshow() and plt.show()
    triangulate(mtx1, mtx2, R, T)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute()
