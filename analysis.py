# Import required modules
import constants as cst
import shutil
import os
import cv2
import math
import numpy as np
from matplotlib import pyplot as plt
import utilities as ut
from moviepy.editor import VideoFileClip
from FeatureMatcher import FeatureMatcher


# video_static: first VideoFileClip
# video_moving: second VideoFileClip
# return: video static offset, video moving offset
def get_audio_offset(video_static, video_moving):
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


# generate undistorted frames images from the video
# video_path: path where to get the video
# calibration_file_path: path to the intrinsics calibration file
# tot_frames: total number of frames
# n_frames: number of frames to generate
# offset: starting offset for the video
# dir_name: directory name where to save the frames
def generate_video_frames(video_path, calibration_file_path, tot_frames, n_frames=30, offset=0, dir_name='sample'):
    SAVE_PATH = cst.FRAMES_FOLDER_PATH + "/" + dir_name

    if not os.path.isfile(calibration_file_path):
        raise Exception('intrinsics file not found!')
    else:
        # Read intrinsics to file
        Kfile = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
        matrix = Kfile.getNode("K").mat()
        distortion = Kfile.getNode("distortion").mat()

    # create BASE_SAVE_PATH folder if not exists
    if not os.path.exists(cst.FRAMES_FOLDER_PATH):
        os.mkdir(cst.FRAMES_FOLDER_PATH)

    # delete previous saved frames images, otherwise create SAVE_PATH folder
    if os.path.exists(SAVE_PATH):
        try:
            shutil.rmtree(SAVE_PATH)
            os.mkdir(SAVE_PATH)
        except OSError as e:
            print("Error: %s - %s." % (e.filename, e.strerror))
    else:
        os.mkdir(SAVE_PATH)

    # Opens the Video file
    video = cv2.VideoCapture(video_path)

    offset *= 30  # 30 fps * offset
    frame_n = offset  # starting from offset (for video sync)
    frame_skip = math.trunc((tot_frames - offset) / n_frames)  # skip threshold between frames to obtain n_frames
    print('Generating frames.. \n')
    for i in range(0, n_frames):
        frame_n += frame_skip
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_n)  # skip frames
        ret, frame = video.read()
        if ret == False:
            raise Exception('Null frame')

        frame_new = ut.undistort_image(frame, matrix, distortion)
        # save image
        cv2.imwrite(SAVE_PATH + '/frame_' + str(i) + '.png', frame_new)
        # cv2.imshow('frame_new', frame_new)
        # cv2.waitKey(0)
    print('Generated frames into \"{}\".. \n'.format(SAVE_PATH))

    video.release()
    cv2.destroyAllWindows()

    return SAVE_PATH


# synchronize the videos and generate undistorted frames of synchronized videos
def sync_videos(video_static_path, video_moving_path):
    if not os.path.isfile(video_static_path):
        raise Exception('Video static not found!')
    if not os.path.isfile(video_moving_path):
        raise Exception('Video dynamic not found!')

    video_static = VideoFileClip(video_static_path)
    video_moving = VideoFileClip(video_moving_path)

    # get offset of the two video
    video_static_offset, video_moving_offset = get_audio_offset(video_static, video_moving)

    # extract filename from video path in order to create a directory for video frames
    video_static_dir = 'static_' + os.path.splitext(os.path.basename(video_static_path))[0]
    video_moving_dir = 'moving_' + os.path.splitext(os.path.basename(video_moving_path))[0]

    # total number of frames of shortest video
    tot_frames = min(ut.get_video_total_frames(video_static_path), ut.get_video_total_frames(video_moving_path))

    # generate frames for static video
    generate_video_frames(video_path=video_static_path,
                          calibration_file_path=cst.INTRINSICS_STATIC_PATH,
                          n_frames=60,
                          tot_frames=tot_frames,
                          dir_name=video_static_dir,
                          offset=video_static_offset)

    # generate frames for moving video
    generate_video_frames(video_path=video_moving_path,
                          calibration_file_path=cst.INTRINSICS_MOVING_PATH,
                          n_frames=60,
                          tot_frames=tot_frames,
                          dir_name=video_moving_dir,
                          offset=video_moving_offset)


def compute(sync=False):
    coin = 'coin1'
    video_static_path = cst.ASSETS_STATIC_FOLDER + '/{}.mov'.format(coin)
    video_moving_path = cst.ASSETS_MOVING_FOLDER + '/{}.mp4'.format(coin)
    frames_static_folder = cst.FRAMES_FOLDER_PATH + '/static_{}'.format(coin)
    frames_moving_folder = cst.FRAMES_FOLDER_PATH + '/moving_{}'.format(coin)

    if sync:
        sync_videos(video_static_path, video_moving_path)

    fm = FeatureMatcher(frames_static_folder, frames_moving_folder,
                        detector_algorithm=FeatureMatcher.DETECTOR_ALGORITHM_ORB,
                        matching_algorithm=FeatureMatcher.MATCHING_ALGORITHM_BRUTEFORCE)
    fm.setKNNTreshold()
    fm.extract_features(show_images=True, save_images=False, plot_histogram=False)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    compute()
