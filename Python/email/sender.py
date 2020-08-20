from smtplib import SMTP, SMTPAuthenticationError, SMTPException

host = "smtp.gmail.com"
port = 587
username = "hungrypy@gmail.com"
password = "iamhungry2016"
from_email = username

def sendmail(to_list, message):
    try:
        email_conn = SMTP(host, port)
        email_conn.ehlo()
        email_conn.starttls()
        email_conn.login(username, password)
        email_conn.sendmail(from_email, to_list,  message)
        email_conn.quit()
    except SMTPAuthenticationError:
        print("could not authenticate")
    except SMTPException:
        print("error: SMTPException")
    except:
        print("something went wrong")