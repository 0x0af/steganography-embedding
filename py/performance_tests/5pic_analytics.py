import os

from PIL import Image

from py.performance_tests.analysis_time import analyze_image
from py.performance_tests.send_report import send_positive_report, send_issue_report

for root, dirs, filenames in os.walk('/home/ftpman'):
    for f in filenames:
        try:
            analyze_image(f, Image.open('/home/ftpman/' + f), 512)
            send_positive_report(f)
        except:
            send_issue_report(f)
