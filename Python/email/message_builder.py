from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText


def buildMessage(from_email, subject, plain_text_msg, html_text_msg):
    the_message = MIMEMultipart("alternative")
    the_message["From"] = from_email
    the_message["Subject"] = subject
    part_text = MIMEText(plain_text_msg, "plain")
    part_html = MIMEText(html_text_msg, "html")
    the_message.attach(part_text)
    the_message.attach(part_html)
    return the_message.as_string()
