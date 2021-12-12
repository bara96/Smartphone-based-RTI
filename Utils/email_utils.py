from Utils import utilities as ut
import smtplib
import ssl
import os
from dotenv import load_dotenv
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart


def send_email(receiver_email, message_subject, message_txt):
    """
    Send an email
    :param receiver_email:
    :param message_subject:
    :param message_txt:
    :return:
    """
    load_dotenv()

    EMAIL_USER = os.getenv('EMAIL_USER')
    EMAIL_PSW = os.getenv('EMAIL_PSW')

    if EMAIL_USER is None or EMAIL_PSW is None:
        ut.console_log("Cannot read .env values", 'red')
        return False

    port = 465  # For SSL

    # Create a secure SSL context
    context = ssl.create_default_context()

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", port, context=context) as server:
            server.login(EMAIL_USER, EMAIL_PSW)
            message = MIMEMultipart("alternative")
            message["Subject"] = message_subject
            message["From"] = EMAIL_USER
            message["To"] = receiver_email
            message.attach(MIMEText(message_txt, "plain"))
            server.sendmail(EMAIL_USER, receiver_email, message.as_string())
            ut.console_log("Email send", 'yellow')
            return True
    except:
        ut.console_log("Error sending email", 'red')
        return False
