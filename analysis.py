# Import required modules
import camera_calibrator
import shutil
import os
import cv2
import math
import utilities as ut
from moviepy.editor import VideoFileClip

FRAMES_FOLDER_PATH = 'assets/frames'


# undistort the image
def undistort_image(image, matrix, distortion):
    # Compute the undistorted image
    h, w = image.shape[:2]
    # Compute the newcamera intrinsic matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 0)
    # Undistort image
    image_new = cv2.undistort(image, matrix, distortion, None, new_camera_matrix)

    return image_new, roi


# generate undistorted frames images from the video
# video_path: path where to get the video
# n_frames: number of frames to generate
# offset: starting offset for the video
# dir_name: directory name where to save the frames
def generate_video_frames(video_path, calibration_file_path, n_frames=30, offset=0, dir_name='sample'):
    SAVE_PATH = FRAMES_FOLDER_PATH + "/" + dir_name

    if not os.path.isfile(calibration_file_path):
        raise Exception('intrinsics file not found!')
    else:
        # Read intrinsics to file
        Kfile = cv2.FileStorage(calibration_file_path, cv2.FILE_STORAGE_READ)
        matrix = Kfile.getNode("K").mat()
        distortion = Kfile.getNode("distortion").mat()

    # create BASE_SAVE_PATH folder if not exists
    if not os.path.exists(FRAMES_FOLDER_PATH):
        os.mkdir(FRAMES_FOLDER_PATH)

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
    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    frame_n = offset  # starting offset (from video sync)
    frame_skip = math.trunc((tot_frames - offset) / n_frames)  # skip threshold between frames to obtain n_frames
    print('Generating frames.. \n')
    for i in range(0, n_frames):
        frame_n += frame_skip
        video.set(1, frame_n)  # skip frames
        ret, frame = video.read()
        if ret == False:
            raise Exception('Null frame')

        frame_new, roi = undistort_image(frame, matrix, distortion)
        # save image
        cv2.imwrite(SAVE_PATH + '/frame_' + str(i) + '.png', frame_new)
        # cv2.imshow('frame', frame_new)
        # cv2.waitKey(0)
    print('Generated frames into \"{}\".. \n'.format(SAVE_PATH))

    video.release()
    cv2.destroyAllWindows()


# video1: first VideoFileClip
# video2: second VideoFileClip
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

    # return the offset
    if swapped:
        return offset, 0  # offset is on static video
    else:
        return 0, offset  # offset is on moving video


# synchronize the videos and generate undistorted frames of synchronized videos
def sync_videos(video_static_path, video_moving_path):
    if not os.path.isfile(video_static_path):
        raise Exception('Video static not found!')
    if not os.path.isfile(video_moving_path):
        raise Exception('Video dynamic not found!')

    video_static = VideoFileClip(video_static_path)
    video_moving = VideoFileClip(video_moving_path)

    # get offset of the two video
    return get_audio_offset(video_static, video_moving)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video_static_path = camera_calibrator.ASSETS_STATIC_FOLDER + '/coin1.mov'
    video_moving_path = camera_calibrator.ASSETS_MOVING_FOLDER + '/coin1.mp4'

    video_static_offset, video_moving_offset = sync_videos(video_static_path, video_moving_path)

    # extract filename from video path in order to create a directory for video frames
    video_static_dir = 'static_' + os.path.splitext(os.path.basename(video_static_path))[0]
    video_moving_dir = 'moving_' + os.path.splitext(os.path.basename(video_moving_path))[0]

    # generate frames for static video
    generate_video_frames(video_static_path,
                          calibration_file_path=camera_calibrator.INTRINSICS_STATIC_PATH,
                          dir_name=video_static_dir,
                          offset=video_static_offset)

    # generate frames for moving video
    generate_video_frames(video_moving_path,
                          calibration_file_path=camera_calibrator.INTRINSICS_MOVING_PATH,
                          dir_name=video_moving_dir,
                          offset=video_moving_offset)
