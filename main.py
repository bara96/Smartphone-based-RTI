import camera_calibrator
import analysis
import interactive_relighting


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    video = 'coin1'
    camera_calibrator.compute()
    analysis.compute(video_name=video, from_storage=False)
    interactive_relighting.compute(video_name=video)
