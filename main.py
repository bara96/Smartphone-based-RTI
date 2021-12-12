import camera_calibrator
import analysis
import interactive_relighting
from Utils import email_utils as eut


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    eut.send_email(receiver_email="matteo.baratella96@gmail.com",
                   message_subject="RTI Notification",
                   message_txt="Interpolation finished")
    '''
    camera_calibrator.compute()
    analysis.compute(video_name='coin1', from_storage=True)
    interactive_relighting.compute()
    '''
