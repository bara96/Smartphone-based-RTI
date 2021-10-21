import glob
import cv2
import os
import numpy as np
from ChessboardDetector import ChessboardDetector


def calibrate_camera(imgdir):
    images = glob.glob('%s/img*.png' % imgdir)
    cd = ChessboardDetector()

    all_corners = []
    imgsize = (0, 0)
    for image in images:
        I = cv2.imread(image)
        imgsize = (I.shape[1], I.shape[0])

        print("Processing %s" % image)
        if cd(I):
            print("3D chessboard points:")
            print(cd.model_points)
            print("2D image points:")
            print(cd.corners)
            cv2.imshow('dbg', cd.dbg_img)
            cv2.waitKey(0)
            head, tail = os.path.split(image)
            cv2.imwrite("dbg/%s" % tail, cd.dbg_img)
            all_corners.append(cd.corners)
            print("Chessboard found")

    # cv calibratecamera requires
    #     object_points as  Nimages x Npoints x 1 x 3
    #     image_points  as  Nimages x Npoints x 1 x 2
    #
    #  type float32
    #

    object_points = np.ascontiguousarray( \
        np.tile(np.expand_dims(np.transpose(cd.model_points),
                               axis=1), (len(all_corners), 1, 1, 1)), dtype=np.float32
    )

    image_points = np.zeros((len(all_corners), cd.model_points.shape[1], 1, 2), dtype=np.float32)

    # Fill image_points
    for (idx, corners) in enumerate(all_corners):
        image_points[idx, :, :, :] = np.ascontiguousarray( \
            np.expand_dims(np.transpose(corners),
                           axis=1), dtype=np.float32
        )

    # Calibrate
    print("\n\n")
    print("Calibrating...")
    (ret, K, dist, rvecs, tvecs) = cv2.calibrateCamera(object_points, image_points, imgsize, None, None)

    print("Calibration result: ")
    print("  RMS: %f" % ret)
    print("    K: \n", K)
    print(" dist: \n", dist)

    # Write instrinsics to file
    Kfile = cv2.FileStorage(imgdir+'/intrinsics.xml', cv2.FILE_STORAGE_WRITE)
    Kfile.write("K", K)
    Kfile.write("dist", dist)


if __name__ == '__main__':
    # calibrate_camera( './snaps/calib/' )
    calibrate_camera('samples/G3DCV2020_data_part1_calibration/calib')
