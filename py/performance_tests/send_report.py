#!/usr/bin/python2.7
import smtplib
import email
import os
from os.path import basename
from email.mime.application import MIMEApplication
from email.MIMEMultipart import MIMEMultipart
from email.Utils import COMMASPACE
from email.MIMEBase import MIMEBase
from email.parser import Parser
from email.MIMEImage import MIMEImage
from email.MIMEText import MIMEText
from email.MIMEAudio import MIMEAudio
import mimetypes
import datetime

def send_report(filename):

 host = 'smtp.gmail.com'
 port = 587
 login = 'antonf.vit@gmail.com'
 password = 'hksiiovmpbwfylzl'
 
 server = smtplib.SMTP()
 server.connect(host,port)
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
 
 file = open(filename, "rb") 
 
 msg.attach(MIMEApplication(
                file.read(),
                Content_Disposition='attachment; filename="%s"' % basename(filename),
                Name=basename(filename)
            ))

 server.sendmail(login,tolist,msg.as_string())
 server.quit()
