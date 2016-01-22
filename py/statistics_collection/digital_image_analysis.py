import getopt
import sys
import traceback

from PIL import Image

from py.multifractal_analysis.mfdfa_di_analysis import analyze_image
from py.statistics_collection.send_report import send_positive_report, send_issue_report

OPTIMAL_SKIP = 16


def analyze_dir(directory, start_index, stop_index):
    filenames = {}

    for i in range(start_index, stop_index + 1, 1):
        filenames[i] = 'im' + str(i) + '.jpg'

    for f in filenames.itervalues():
        try:
            analyze_image(f, Image.open(directory + '\\' + f), OPTIMAL_SKIP)
            send_positive_report(f + '.mat')
        except Exception, e:
            send_issue_report(f, e, repr(traceback.format_exc()))


def main(argv):
    directory = ''
    start_index = 1
    stop_index = 25000

    try:
        opts, args = getopt.getopt(argv, "",
                                   ["directory=", "start_index=", "stop_index="])
    except getopt.GetoptError:
        print '\r\nPlease, use this software this way:'
        print 'Embedding: digital_image_analysis.py --directory %PATH_TO_DIGITAL_IMAGES% --start_index %start_index% ' \
              '--stop_index %stop_index%'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--directory':
            directory = arg
        if opt == '--start_index':
            start_index = int(arg)
        if opt == '--stop_index':
            stop_index = int(arg)

    analyze_dir(directory, start_index, stop_index)


if __name__ == "__main__":
    main(sys.argv[1:])
