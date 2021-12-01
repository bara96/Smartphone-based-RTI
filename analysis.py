# Import required modules
import constants as cst
import os
import cv2
import math
import utilities as ut
from moviepy.editor import VideoFileClip
from FeatureMatcher import FeatureMatcher


def get_audio_offset(video_static, video_moving):
    """
    Return video static offset, video moving offset
    :param video_static: first VideoFileClip
    :param video_moving: second VideoFileClip
    :return:
    """
    video1 = video_static
    swapped = False
    video2 = video_moving

    # swap the videos if the second one is greater
    if video1.duration > video2.duration:
        swapped = True
        video = video1
        video1 = video2
        video2 = video

    # extract audio from videos
    audio1 = video1.audio
    audio1_path = os.path.splitext(video1.filename)[0] + ".wav"
    audio1.write_audiofile(audio1_path)

    audio2 = video2.audio
    audio2_path = os.path.splitext(video2.filename)[0] + ".wav"
    audio2.write_audiofile(audio2_path)

    # get the video shift
    fs1, wave1 = ut.stereo_to_mono_wave(audio1_path)
    fs2, wave2 = ut.stereo_to_mono_wave(audio2_path)

    # calculate and round the shift
    offset = round(ut.find_audio_correlation(wave2, wave1) / fs1, 0)

    print("Current offset: {} \n".format(offset))

    # return the offset
    if swapped:
        return offset, 0  # offset is on static video
    else:
        return 0, offset  # offset is on moving video


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
        return True

    matrix, distortion = ut.get_camera_intrinsics(calibration_file_path)

    # Opens the Video file
    video = cv2.VideoCapture(video_path)
    default_frame = None
    lower_brightness = 1
    for i in range(0, 90):
        ret, frame = video.read()
        if not ret:
            raise Exception('Null frame')

        frame_new = ut.undistort_image(frame, matrix, distortion)
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
    return True


def extract_video_frames(static_video_path, moving_video_path, tot_frames, max_frames=0,
                         video_static_offset=0, video_moving_offset=0, start_from_frame=0):
    """
    Get undistorted frames images from the video_static and extract features
    :param static_video_path: path to the video_static
    :param moving_video_path: path to the video_moving
    :param tot_frames: total number of frames of the video_static
    :param max_frames: set a max n° of frames to read
    :param video_static_offset: starting offset for the static video
    :param video_moving_offset: starting offset for the moving video
    :param start_from_frame: starting a given frame
    :return:
    """

    matrix_static, distortion_static = ut.get_camera_intrinsics(cst.INTRINSICS_STATIC_PATH)
    matrix_moving, distortion_moving = ut.get_camera_intrinsics(cst.INTRINSICS_MOVING_PATH)

    # Opens the Video file
    video_static = cv2.VideoCapture(static_video_path)
    video_moving = cv2.VideoCapture(moving_video_path)

    video_static_offset *= 30  # 30 fps * offset
    video_moving_offset *= 30  # 30 fps * offset

    # starting from offset (for video_static sync)
    frame_static_fps_count = 0
    frame_static_cursor = video_static_offset
    frame_moving_cursor = video_moving_offset

    # skip threshold between frames to read a maximum of max_frames
    if max_frames is not 0:
        offset = max(video_static_offset, video_moving_offset)
        if max_frames < int(tot_frames/8):
            # keep at least 1/8 of the frames for a good precision
            max_frames = int(tot_frames/8)
        if max_frames > tot_frames:
            max_frames = tot_frames - offset
        frame_skip = math.trunc((tot_frames - offset) / max_frames)
    else:
        offset = max(video_static_offset, video_moving_offset)
        max_frames = tot_frames - offset
        frame_skip = 1

    print("Max frames to read:", max_frames, "\n")

    if 0 < start_from_frame < tot_frames:
        frame_static_cursor += start_from_frame * frame_skip
        frame_moving_cursor += start_from_frame * frame_skip
        frame_static_fps_count = int(frame_moving_cursor / 25)
        frame_static_cursor += frame_static_fps_count
        video_static.set(cv2.CAP_PROP_POS_FRAMES, frame_static_cursor)
        video_moving.set(cv2.CAP_PROP_POS_FRAMES, frame_moving_cursor)

    fm = FeatureMatcher()
    # set show parameters for visual debug information
    fm.setShowParams(show_static_frame=True, show_moving_frame=True,
                     show_rectangle_canvas=True, show_corners=True,
                     show_homography=False, show_light_direction=True)

    dataset = []
    for i in range(start_from_frame, max_frames - 1):
        print("Frame n° ", i)
        ret_static, frame_static = video_static.read()
        ret_moving, frame_moving = video_moving.read()
        if ret_static is False or ret_moving is False:
            ut.console_log('Error: Null frame', 'e')
            continue

        frame_static = ut.undistort_image(frame_static, matrix_static, distortion_static)
        frame_moving = ut.undistort_image(frame_moving, matrix_moving, distortion_moving)
        static_shape, static_shape_points = fm.computeStaticShape(frame_static)
        if static_shape_points is not None:
            result = fm.extractFeatures(moving_img=frame_moving, static_img=static_shape,
                                        static_shape_points=static_shape_points,
                                        wait_key=False)
            if result is not False:
                dataset.append(result)

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


def sync_videos(video_static_path, video_moving_path):
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
    video_static_offset, video_moving_offset = get_audio_offset(video_static, video_moving)

    # extract filename from video path in order to create a directory for video frames
    video_default_frame = 'default_' + os.path.splitext(os.path.basename(video_static_path))[0]

    # total number of frames of shortest video
    tot_frames = min(ut.get_video_total_frames(video_static_path), ut.get_video_total_frames(video_moving_path))

    # generate default frame from static video
    generate_video_default_frame(video_path=video_static_path,
                                 calibration_file_path=cst.INTRINSICS_STATIC_PATH,
                                 file_name=video_default_frame)

    return video_static_offset, video_moving_offset, tot_frames


def compute(video_name='coin1'):
    """
    Main function
    :param video_name: name of the video to take
    """
    video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(video_name)
    video_moving_path = cst.ASSETS_MOVING_FOLDER + '/{}.mp4'.format(video_name)

    # extract features directly from video, without saving frame images
    video_static_offset, video_moving_offset, tot_frames = sync_videos(video_static_path, video_moving_path)
    results = extract_video_frames(static_video_path=video_static_path,
                                   moving_video_path=video_moving_path,
                                   tot_frames=tot_frames,
                                   max_frames=300,
                                   video_moving_offset=video_moving_offset,
                                   video_static_offset=video_static_offset,
                                   start_from_frame=0)

    # write results on file
    file_path = "assets/results_{}.pickle".format(video_name)
    ut.write_on_file(results, file_path)

    return results


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute(video_name='coin1')
