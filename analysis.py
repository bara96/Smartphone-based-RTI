# Import required modules
import numpy as np

import constants as cst
from Utils import audio_utils as aut
from Utils import image_utils as iut
from Utils import utilities as ut
from FeatureMatcher import FeatureMatcher
import os
import cv2
import math
import matplotlib.pyplot as plt


def generate_video_default_frame(video_path, calibration_file_path, file_name='default'):
    """
    Generate default video frame (with no light)
    :param video_path: path to the video
    :param calibration_file_path: path to the intrinsics calibration file
    :param file_name: file name to save the frame
    """
    SAVE_PATH = "assets/" + file_name + ".png"
    if os.path.isfile(SAVE_PATH):
        print("Default frame already exists \n")
        return SAVE_PATH

    matrix, distortion = ut.get_camera_intrinsics(calibration_file_path)

    # Opens the Video file
    video = cv2.VideoCapture(video_path)
    default_frame = None
    lower_brightness = 1
    for i in range(0, 90):
        ret, frame = video.read()
        if not ret:
            raise Exception('Null frame')

        frame_new = iut.undistort_image(frame, matrix, distortion)
        mean_hv_ls = cv2.mean(cv2.cvtColor(frame_new, cv2.COLOR_BGR2HSV))
        mean_brightness = mean_hv_ls[2] / 255
        # set as default image the one with brightness near 0.5
        if lower_brightness > mean_brightness > 0.5:
            default_frame = frame_new.copy()
            lower_brightness = mean_brightness

    video.release()
    # cv2.imshow('Default', default_frame)
    # cv2.waitKey(0)
    cv2.imwrite(SAVE_PATH, default_frame)
    cv2.destroyAllWindows()
    print("Default frame generated \n")
    return SAVE_PATH


def sync_videos(video_static_path, video_moving_path):
    from moviepy.editor import VideoFileClip
    """
    Synchronize the videos and get the offset and n° of frames
    :param video_static_path: path to the static video
    :param video_moving_path: path to the moving video
    """
    if not os.path.isfile(video_static_path):
        raise Exception('Video static not found!')
    if not os.path.isfile(video_moving_path):
        raise Exception('Video dynamic not found!')

    video_static = VideoFileClip(video_static_path)
    video_moving = VideoFileClip(video_moving_path)

    # get offset of the two video
    video_static_offset, video_moving_offset = aut.get_audio_offset(video_static, video_moving)

    # total number of frames of shortest video
    tot_frames = min(iut.get_video_total_frames(video_static_path), iut.get_video_total_frames(video_moving_path))

    return video_static_offset, video_moving_offset, tot_frames


def extract_video_frames(static_video_path, moving_video_path,
                         tot_frames, max_frames=0, start_from_frame=0,
                         video_static_offset=0, video_moving_offset=0,
                         default_frame_path="default.png"):
    global x_plot
    global y_plot
    """
    Get undistorted frames images from the video_static and extract features
    :param static_video_path: path to the video_static
    :param moving_video_path: path to the video_moving
    :param tot_frames: total number of frames of the video_static
    :param max_frames: set a max n° of frames to read
    :param video_static_offset: starting offset for the static video
    :param video_moving_offset: starting offset for the moving video
    :param start_from_frame: starting a given frame
    :param default_frame_path: name of the default frame file
    :return:
    """

    matrix_static, distortion_static = ut.get_camera_intrinsics(cst.INTRINSICS_STATIC_PATH)
    matrix_moving, distortion_moving = ut.get_camera_intrinsics(cst.INTRINSICS_MOVING_PATH)

    # Opens the Video file
    video_static = cv2.VideoCapture(static_video_path)
    video_moving = cv2.VideoCapture(moving_video_path)

    video_static_offset = round(video_static_offset * 30, 0)  # 30 fps * offset
    video_moving_offset = round(video_moving_offset * 30, 0)  # 30 fps * offset

    # starting from offset (for video_static sync)
    frame_static_fps_count = 0
    frame_static_cursor = video_static_offset  # skip offset
    frame_moving_cursor = video_moving_offset  # skip offset

    # skip two seconds
    if video_moving_offset < 1.5 or video_static_offset < 1.5:
        frame_static_cursor += 60
        frame_moving_cursor += 60

    # skip threshold between frames to read a maximum of max_frames
    if max_frames is not 0:
        offset = max(video_static_offset, video_moving_offset)
        if max_frames < int(tot_frames / 8):
            # keep at least 1/8 of the frames for a good precision
            max_frames = int(tot_frames / 8)
        if max_frames > tot_frames:
            max_frames = tot_frames - offset
        frame_skip = math.trunc((tot_frames - offset) / max_frames)
    else:
        offset = max(video_static_offset, video_moving_offset)
        max_frames = tot_frames - offset
        frame_skip = 1

    print("Max frames to read:", max_frames)

    if 0 < start_from_frame < tot_frames:
        frame_static_cursor += start_from_frame * frame_skip
        frame_moving_cursor += start_from_frame * frame_skip
        frame_static_fps_count = int(frame_moving_cursor / 25)
        frame_static_cursor += frame_static_fps_count
        video_static.set(cv2.CAP_PROP_POS_FRAMES, frame_static_cursor)
        video_moving.set(cv2.CAP_PROP_POS_FRAMES, frame_moving_cursor)

    print("Static video starting frame:", frame_static_cursor)
    print("Moving video starting frame:", frame_moving_cursor)

    fm = FeatureMatcher()
    # set show parameters for visual debug information
    fm.setShowParams(show_static_frame=True, show_moving_frame=True,
                     show_rectangle_canvas=True, show_corners=True,
                     show_homography=False, show_light_direction=True,
                     debug=False)

    # compute static shape detection only on default frame, since they've all the same homography
    frame_default = cv2.imread(default_frame_path)
    static_shape_cnts, static_shape_points = fm.computeStaticShape(frame_default)

    dataset = []
    failures_consecutive_count = 0
    for i in range(start_from_frame, max_frames - 1):
        print("Frame n° ", i)
        ret_static, frame_static = video_static.read()
        ret_moving, frame_moving = video_moving.read()
        if ret_static is False or ret_moving is False:
            ut.console_log('Error: Null frame')
            continue

        frame_static = iut.undistort_image(frame_static, matrix_static, distortion_static)
        frame_moving = iut.undistort_image(frame_moving, matrix_moving, distortion_moving)
        result = fm.extractFeatures(moving_img=frame_moving, static_img=frame_static,
                                    static_shape_points=static_shape_points, static_shape_cnts=static_shape_cnts,
                                    wait_key=False)
        if result is not False:
            dataset.append(result)
            failures_consecutive_count = 0
        else:
            failures_consecutive_count += 1
            if failures_consecutive_count > 5:
                fm.resetPreviousCorners()
                failures_consecutive_count = 0

        # every 25 frames skip a frame of the static video to keep sync
        '''
        if frame_moving_cursor > 0:
            fps_count = int(frame_moving_cursor / 25)
            if fps_count > frame_static_fps_count:
                frame_static_fps_count = fps_count
                frame_static_cursor += 1
        '''

        # skip frames
        frame_static_cursor += frame_skip
        frame_moving_cursor += frame_skip
        video_static.set(cv2.CAP_PROP_POS_FRAMES, frame_static_cursor)
        video_moving.set(cv2.CAP_PROP_POS_FRAMES, frame_moving_cursor)

    video_static.release()
    video_moving.release()
    cv2.destroyAllWindows()

    return dataset


def compute_intensities(data, show_pixel=False):
    """
    Compute light vectors intensities foreach frame pixel
    :param data: array of tuples (intensities, camera_position), for each frame
    intensities: array of intensities for each pixel of the ROI, for the current frame
    camera_position: tuple (x, y, z), for the current frame
    :param show_pixel: if True, show first pixel light vectors values
    :rtype: object
    """
    if data is None or len(data) <= 0:
        ut.console_log("Error computing intensities: results are empty")

    print("Computing intensities values")

    pixels_lx = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]
    pixels_ly = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]
    pixels_intensity = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]

    i = 0
    for frame_data in data:
        print("Frame n° ", i)
        i += 1
        intensities = frame_data[0]
        camera_position = frame_data[1]
        for y in range(cst.ROI_DIAMETER):
            for x in range(cst.ROI_DIAMETER):
                p = (x, y, 0)
                l = (camera_position - p) / ut.euclidean_distance(camera_position[0], camera_position[1], p[0], p[1])
                pixels_lx[y][x].append(l[0])
                pixels_ly[y][x].append(l[1])
                pixels_intensity[y][x].append(intensities[y][x])

    pixels_lx = np.around(pixels_lx, decimals=2)
    pixels_ly = np.around(pixels_ly, decimals=2)
    pixels_intensity = np.array(pixels_intensity)

    if show_pixel:
        # extract first pixel values for plot
        lx = pixels_lx[0][0]
        ly = pixels_ly[0][0]
        val = pixels_intensity[0][0]

        print("lx", lx)
        print("ly", ly)
        print("val", val)

        plt.scatter(lx, ly, c=val)
        plt.show()

    pixels_data = (pixels_lx, pixels_ly, pixels_intensity)
    return np.array(pixels_data)


def interpolate_intensities(data, show_pixel=False):
    from scipy.interpolate import Rbf
    """
    Interpolate pixel intensities
    :param data: array of tuples (pixels_lx, pixels_ly, pixels_intensity), for each pixel
    pixels_lx: list of lx coordinates for each value, for the current pixel
    pixels_ly: list of ly coordinates for each value, for the current pixel
    pixels_intensity: list of intensities, for current pixel
    :param show_pixel: if True, show first pixel interpolation
    """
    if data is None or len(data) <= 0:
        ut.console_log("Error computing interpolation: results are empty")

    print("Computing interpolation values")

    interpolated_lx = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]
    interpolated_ly = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]
    interpolated_intensity = [[[] for y in range(cst.ROI_DIAMETER)] for x in range(cst.ROI_DIAMETER)]

    pixels_lx = data[0]
    pixels_ly = data[1]
    pixels_intensity = data[2]
    i = 0
    for y in range(0, 1):
        for x in range(0, 1):
            print("Frame n° ", i)
            i += 1
            lx = pixels_lx[y][x]
            ly = pixels_ly[y][x]
            lz = np.ones(len(lx))
            val = pixels_intensity[y][x]

            rbfi = Rbf(lx, ly, lz, val)  # radial basis function interpolator instance

            xi = yi = zi = np.linspace(-1, 1, 1000)
            di = rbfi(xi, yi, zi)  # interpolated values

            interpolated_lx[y][x] = xi
            interpolated_ly[y][x] = yi
            interpolated_intensity[y][x] = di

    interpolated_lx = np.around(interpolated_lx, decimals=2)
    interpolated_ly = np.around(interpolated_ly, decimals=2)

    if show_pixel:
        # plot only first pixel values
        lx = interpolated_lx[0][0]
        ly = interpolated_ly[0][0]
        val = interpolated_intensity[0][0]

        print("interpolated_lx", lx)
        print("interpolated_ly", ly)
        print("interpolated_val", val)
        plt.scatter(lx, ly, c=val)
        plt.show()

    interpolated_data = (interpolated_lx, interpolated_ly, interpolated_intensity)
    return interpolated_data


def compute(video_name='coin1', from_storage=False, storage_filepath=None):
    """
    Main function
    :param video_name: name of the video to take
    :param from_storage: if True read results from a saved file, otherwise compute results from skratch
    :param storage_filepath:  if None is set read results from default filepath, otherwise it must be a filepath to a valid results file
    """

    results_file_path = "assets/results_{}.pickle".format(video_name)

    ut.console_log("Step 1: Computing frames values \n", 'blue')
    if from_storage is True:
        # read a pre-saved results file
        if storage_filepath is not None:
            results_file_path = from_storage
        if not os.path.isfile(results_file_path):
            raise Exception('Storage results file not found!')
        print("Reading values from storage")
        results = ut.read_from_file(results_file_path)
    else:
        # compute results from skratch
        print("Generating frames values")
        video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(video_name)
        video_moving_path = cst.ASSETS_MOVING_FOLDER + '/{}.mp4'.format(video_name)

        # extract features directly from video, without saving frame images
        video_static_offset, video_moving_offset, tot_frames = sync_videos(video_static_path, video_moving_path)

        # extract filename from video path in order to create a directory for video frames
        default_frame_name = 'default_' + os.path.splitext(os.path.basename(video_static_path))[0]
        # generate default frame from static video
        default_frame_path = generate_video_default_frame(video_path=video_static_path,
                                                          calibration_file_path=cst.INTRINSICS_STATIC_PATH,
                                                          file_name=default_frame_name)

        results = extract_video_frames(static_video_path=video_static_path,
                                       moving_video_path=video_moving_path,
                                       tot_frames=tot_frames,
                                       max_frames=300,
                                       video_moving_offset=video_moving_offset,
                                       video_static_offset=video_static_offset,
                                       start_from_frame=0,
                                       default_frame_path=default_frame_path)

        # write results on file
        ut.write_on_file(results, results_file_path)

    ut.console_log("Step 2: Computing pixels intensities \n", 'blue')
    # compute light vectors intensities
    data = compute_intensities(results, show_pixel=True)

    ut.console_log("Step 3: Computing interpolation \n", 'blue')
    # interpolate pixel intensities
    interpolate_intensities(data, show_pixel=True)


def test():
    from scipy.interpolate import Rbf

    x_coarse, y_coarse = np.mgrid[-1:1:0.2, -1:1:0.2]
    print("x_coarse", x_coarse)
    x_fine, y_fine = np.mgrid[-1:1:0.01, -1:1:0.01]
    print("x_fine", x_fine)
    data_coarse = np.ones([10, 10])

    rbfi = Rbf(x_coarse.ravel(), y_coarse.ravel(), data_coarse.ravel())

    interpolated_data = rbfi(x_fine.ravel(), y_fine.ravel()).reshape([x_fine.shape[0], y_fine.shape[0]])

    plt.imshow(interpolated_data)
    plt.waitforbuttonpress()

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coin = 1
    storage_results_save = "assets/results_coin{}_save.pickle".format(coin)
    compute(video_name='coin1', from_storage=True)
