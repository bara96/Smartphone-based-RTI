# Import required modules
import camera_calibrator
import os
import cv2
import numpy as np
import subprocess


# extract reference audio
# video_path: video path from where to extract the audio
def extract_audio(video_path):
    if not os.path.isfile(video_path):
        raise Exception('Video not found!')
    audio_path = os.path.splitext(video_path)[0] + ".wav"
    command = "ffmpeg -i \"{0}\" -map 0:1 -acodec pcm_s16le -ac 2 \"{1}\"".format(video_path, audio_path)
    os.system(command)
    print("Extracted: ", audio_path, "\n")
    return audio_path

def sync_videos(video_static_path, video_moving_path):
    if not os.path.isfile(video_static_path):
        raise Exception('Video static not found!')
    if not os.path.isfile(video_moving_path):
        raise Exception('Video dynamic not found!')

    results = []

    video_static_audio = extract_audio(video_static_path)
    video_moving_audio = extract_audio(video_moving_path)

    command = "praatcon crosscorrelate.praat \"{}\" \"{}\"".format(os.path.abspath(video_static_audio), os.path.abspath(video_moving_audio))
    result = subprocess.check_output(command, shell=True)
    print(result)

    clip_start = 133
    in_name = video_static_audio
    out_name = in_name.split('.')[0] + "_part.mp4"
    offset = round(float(result), 3)
    clip_start += offset
    command = "ffmpeg -i \n{0}\n -ss {1} \n{2}\n".format(in_name, str(clip_start), out_name)
    os.system(command)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    dir_static = 'assets/G3DCV2021_data/cam1 - static/coin1.mov'
    dir_moving = 'assets/G3DCV2021_data/cam2 - moving light/coin1.mp4'
    sync_videos(dir_static, dir_moving)
