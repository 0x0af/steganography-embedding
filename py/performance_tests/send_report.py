#!/usr/bin/python2.7
import datetime
import email
import smtplib
from email.mime.text import MIMEText
from email.mime.application import MIMEApplication
from os.path import basename


def send_positive_report(filename):
    host = 'smtp.gmail.com'
    port = 587
    login = 'antonf.vit@gmail.com'
    password = 'hksiiovmpbwfylzl'

    server = smtplib.SMTP()
    server.connect(host, port)
    server.ehlo()
    server.starttls()
    server.login(login, password)

    fromaddr = 'Stego Reporter Bot'
    tolist = '0x0af@ukr.net'
    sub = 'Stego Report ' + datetime.date.today().strftime('%d, %b %Y')
    body = 'New block of image statistics is ready. Please, see the data pinned'

    msg = email.MIMEMultipart.MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = tolist
    msg['Subject'] = sub
    msg.attach(MIMEText(body))
    msg.attach(MIMEText('Best regards,\r\nStego Reporter Bot', 'plain'))

    s_file = open(filename, "rb")

    msg.attach(MIMEApplication(
        s_file.read(),
        Content_Disposition='attachment; filename="%s"' % basename(filename),
        Name=basename(filename)
    ))

    server.sendmail(login, tolist, msg.as_string())
    server.quit()


def send_issue_report(image_filename, e, stacktrace):
    host = 'smtp.gmail.com'
    port = 587
    login = 'antonf.vit@gmail.com'
    password = 'hksiiovmpbwfylzl'

    server = smtplib.SMTP()
    server.connect(host, port)
    server.ehlo()
    server.starttls()
    server.login(login, password)

    fromaddr = 'Stego Reporter Bot'
    tolist = '0x0af@ukr.net'
    sub = 'Stego Report ' + datetime.date.today().strftime('%d, %b %Y')
    body = 'Problem appeared during the workflow (' + e + '), please give some attention.\r\nProblematic picture: ' \
           + image_filename + '\r\nStackTrace: ' + stacktrace

    msg = email.MIMEMultipart.MIMEMultipart()
    msg['From'] = fromaddr
    msg['To'] = tolist
    msg['Subject'] = sub
    msg.attach(MIMEText(body))
    msg.attach(MIMEText('Best regards,\r\nStego Reporter Bot', 'plain'))

    server.sendmail(login, tolist, msg.as_string())
    server.quit()
