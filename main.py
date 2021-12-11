import camera_calibrator
import analysis
import interactive_relighting


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    camera_calibrator.compute()
    analysis.compute(video_name='coin1', from_storage=True)
    interactive_relighting.compute()
