# Import required modules
import numpy as np
import constants as cst
from Utils import audio_utils as aut
from Utils import image_utils as iut
from Utils import email_utils as eut
from Utils import utilities as ut
from FeatureMatcher import FeatureMatcher
from scipy.interpolate import Rbf
import os
import cv2
import math
import matplotlib.pyplot as plt
from tqdm import tqdm
from timeit import default_timer as timer
from numba import numba, jit, cuda


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
                         tot_frames, video_static_offset=0, video_moving_offset=0,
                         default_frame_path="default.png"):
    """
    Get undistorted frames images from the video_static and extract features
    :param static_video_path: path to the video_static
    :param moving_video_path: path to the video_moving
    :param tot_frames: total number of frames of the video_static
    :param video_static_offset: starting offset for the static video
    :param video_moving_offset: starting offset for the moving video
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
    # frame_static_cursor += 60
    # frame_moving_cursor += 60

    start_from_frame = 0  # starting from a given frame
    max_frames_to_read = int(tot_frames / 8)  # set a max n° of frames to read
    offset = max(video_static_offset, video_moving_offset)
    frame_skip = math.trunc((tot_frames - offset) / max_frames_to_read)

    print("Max frames to read:", max_frames_to_read)

    if 0 < start_from_frame < max_frames_to_read:
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
    for i in tqdm(range(start_from_frame, max_frames_to_read - 1)):
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
            if failures_consecutive_count > 4:
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


# @jit(forceobj=True)
def compute_intensities(data, first_only=False):
    """
    Compute light vectors intensities foreach frame pixel
    :param data: array of tuples (intensities, camera_position), for each frame
    intensities: array of intensities for each pixel of the ROI, for the current frame
    camera_position: tuple (x, y, z), for the current frame
    :param first_only: compute and show only first pixel evaluation
    :rtype: object
    """
    if data is None or len(data) <= 0:
        raise Exception("Error computing intensities: results are empty")

    print("Computing intensities values:")

    range_val = cst.ROI_DIAMETER
    if first_only:
        ut.console_log("Intensities of first pixel only", "yellow")
        range_val = 1

    pixels_lx = np.empty((range_val, range_val, len(data)), dtype=np.float32)
    pixels_ly = np.empty((range_val, range_val, len(data)), dtype=np.float32)
    pixels_intensity = np.empty((range_val, range_val, len(data)), dtype=np.int32)

    for i in tqdm(range(len(data))):
        frame_data = data[i]
        intensities = frame_data[0]
        camera_position = frame_data[1]
        for y in range(range_val):
            for x in range(range_val):
                p = (x, y, 0)
                l = (camera_position - p) / np.linalg.norm(camera_position - p)
                pixels_lx[y][x][i] = l[0]
                pixels_ly[y][x][i] = l[1]
                pixels_intensity[y][x][i] = intensities[y][x]

    if first_only:
        # plot only first pixel values
        lx = pixels_lx[0][0]
        ly = pixels_ly[0][0]
        val = pixels_intensity[0][0]

        # print("lx", lx)
        # print("ly", ly)
        # print("val", val)

        plt.scatter(lx, ly, c=val)
        plt.xlabel('lx')
        plt.ylabel('ly')
        plt.show()

    pixels_data = (pixels_lx, pixels_ly, pixels_intensity)
    return pixels_data


# @jit(forceobj=True)
def interpolate_intensities(data, first_only=False):
    """
    Interpolate pixel intensities
    :param data: array of tuples (pixels_lx, pixels_ly, pixels_intensity), for each pixel
    pixels_lx: list of lx coordinates for each value, for the current pixel
    pixels_ly: list of ly coordinates for each value, for the current pixel
    pixels_intensity: list of intensities, for current pixel
    :param first_only: compute and show only first pixel evaluation
    """
    if data is None or len(data) != 3:
        raise Exception("Error computing interpolation: results are empty or invalid")

    print("Computing interpolation values:")

    range_val = cst.ROI_DIAMETER
    if first_only:
        range_val = 1

    pixels_lx = data[0]
    pixels_ly = data[1]
    pixels_intensity = data[2]
    # compute the normalized area domain
    # roi_area_domain = np.linspace(-1.0, 1.0, 200)
    # xi, yi = np.meshgrid(roi_area_domain, roi_area_domain)
    yi, xi = np.mgrid[-1:1:cst.INTERPOLATION_PARAM, -1:1:cst.INTERPOLATION_PARAM]
    yi = np.around(yi, decimals=2)
    xi = np.around(xi, decimals=2)

    interpolated_intensities = [[[] for y in range(range_val)] for x in range(range_val)]
    for y in tqdm(range(range_val)):
        for x in range(range_val):
            lx = pixels_lx[y][x]
            ly = pixels_ly[y][x]
            val = pixels_intensity[y][x]

            rbfi = Rbf(lx, ly, val, function='linear')  # radial basis function interpolator instance

            # interpolated values
            di = rbfi(xi, yi)
            interpolated_intensities[y][x] = di

    if first_only:
        # plot only first pixel values
        val = interpolated_intensities[0][0]

        # print("interpolated_val", val)

        plt.scatter(xi, yi, c=val)
        plt.xlabel('lx')
        plt.ylabel('ly')
        plt.show()

    return interpolated_intensities


def prepare_images_data(data, first_only=False):
    """
    Prepare images for each camera position (ly, lx) with interpolated values
    :param data: list of interpolated values for each pixel (y,x) and each light direction (ly,lx)
    interpolation_intensities[y][x][ly][lx] = intensity
    :param first_only: compute only first pixel evaluation
    :return:
    """

    if data is None or len(data) <= 0:
        raise Exception("Error preparing images: results are empty")

    print("Preparing images values:")

    yi, xi = np.mgrid[-1:1:cst.INTERPOLATION_PARAM, -1:1:cst.INTERPOLATION_PARAM]
    xi = np.around(xi, decimals=2)
    yi = xi[0]
    xi = xi[0]

    range_val = cst.ROI_DIAMETER
    if first_only:
        range_val = 1

    # prepare images for each position
    interpolated_images = [[[] for y in range(len(yi))] for x in range(len(xi))]
    for ly in tqdm(range(len(yi))):
        for lx in range(len(xi)):
            # get image for current light position ly lx
            img = np.empty((range_val, range_val), dtype=np.int32)
            for y in range(range_val):
                for x in range(range_val):
                    img[y][x] = data[y][x][ly][lx]
            interpolated_images[ly][lx] = img

    return interpolated_images


def compute(video_name='coin1', from_storage=False, storage_filepath=None, notification_email=True, debug=False):
    """
    Main function
    :param video_name: name of the video to take
    :param from_storage: if True read results from a saved file, otherwise compute results from skratch
    :param storage_filepath: if None is set read results from default filepath, otherwise it must be a filepath to a valid results file
    :param notification_email: send a notification email when finished
    :param debug: compute a debug run with only the first pixel
    """

    results_frames_filepath = "assets/frames_results_{}".format(video_name)
    results_interpolation_filepath = "assets/interpolation_results_{}".format(video_name)

    ut.console_log("Step 1: Computing frames values", 'blue', newline=True)
    if from_storage is True:
        # read a pre-saved results file
        if storage_filepath is not None:
            results_frames_filepath = storage_filepath
        results_frames = ut.read_from_file(results_frames_filepath)
    else:
        # compute results from skratch
        print("Generating frames values")
        video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(video_name)
        video_moving_path = cst.ASSETS_MOVING_FOLDER + '/{}.mp4'.format(video_name)

        # extract features directly from video, without saving frame images
        video_static_offset, video_moving_offset, tot_frames = sync_videos(video_static_path, video_moving_path)

        # set default frame filename
        default_frame_name = 'default_{}'.format(video_name)
        # generate default frame from static video
        default_frame_path = generate_video_default_frame(video_path=video_static_path,
                                                          calibration_file_path=cst.INTRINSICS_STATIC_PATH,
                                                          file_name=default_frame_name)

        results_frames = extract_video_frames(static_video_path=video_static_path,
                                              moving_video_path=video_moving_path,
                                              tot_frames=tot_frames,
                                              video_moving_offset=video_moving_offset,
                                              video_static_offset=video_static_offset,
                                              default_frame_path=default_frame_path)

        np.array(results_frames)
        # write frames results on file
        ut.write_on_file(results_frames, results_frames_filepath)

    if debug:
        ut.console_log("Notice: computing in debug mode (first pixel only)", "yellow")

    ut.console_log("Step 2: Computing pixels intensities", 'blue', newline=True)
    # compute light vectors intensities
    data = compute_intensities(results_frames, first_only=debug)

    ut.console_log("Step 3: Computing interpolation", 'blue', newline=True)
    # interpolate pixel intensities
    results_interpolation = interpolate_intensities(data, first_only=debug)

    ut.console_log("Step 4: Preparing images data", 'blue', newline=True)
    results_images = prepare_images_data(results_interpolation, first_only=debug)

    if debug is False:
        ut.write_on_file(results_images, results_interpolation_filepath, compressed=False)

        if notification_email:
            eut.send_email(receiver_email="matteo.baratella96@gmail.com",
                           message_subject="RTI Notification",
                           message_txt="Interpolation finished")

    ut.console_log("OK. Computation completed", 'green', newline=True)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    coin = 1
    storage_results_save = "assets/frames_results_coin{}".format(coin)

    start = timer()
    compute(video_name='coin{}'.format(coin), from_storage=False, debug=False)
    time = round(timer() - start, 2)
    minutes = int(time / 60)
    seconds = time - (minutes * 60)
    print("Computation duration: {} m {} s".format(minutes, seconds))
