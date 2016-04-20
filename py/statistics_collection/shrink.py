import getopt
import os
import sys

import numpy
from scipy.io import loadmat, savemat

OPTIMAL_SKIP = 16


def shrink_dir(directory, start_index, stop_index):
    filenames = {}

    for i in range(start_index, stop_index + 1, 1):
        filenames[i * 4] = 'im' + str(i) + '.jpg-c.mat'
        filenames[i * 4 + 1] = 'im' + str(i) + '.jpg-am.mat'
        filenames[i * 4 + 2] = 'im' + str(i) + '.jpg-lm.mat'
        filenames[i * 4 + 3] = 'im' + str(i) + '.jpg-he.mat'

    for f in filenames.itervalues():
        try:
            structure = loadmat(directory + '/' + f)
            spectrum = structure.pop('spectrum')
            shrinked_spectrum = numpy.zeros((3, int(512 / OPTIMAL_SKIP), 4, 300))
            for color_channel in range(0, 3):
                for row_index in range(0, 512, OPTIMAL_SKIP):
                    shrinked_spectrum[color_channel][int(row_index / OPTIMAL_SKIP)][0] = \
                        spectrum[color_channel][row_index][0]
                    shrinked_spectrum[color_channel][int(row_index / OPTIMAL_SKIP)][1] = \
                        spectrum[color_channel][row_index][1]
                    shrinked_spectrum[color_channel][int(row_index / OPTIMAL_SKIP)][2] = \
                        spectrum[color_channel][row_index][2]
                    shrinked_spectrum[color_channel][int(row_index / OPTIMAL_SKIP)][3] = \
                        spectrum[color_channel][row_index][3]
            structure['spectrum'] = shrinked_spectrum
            savemat(directory + '/' + f, structure)
        except Exception, e:
            print e


def removekey(d, key):
    r = dict(d)
    del r[key]
    return r


def main(argv):
    directory = ''
    start_index = 1
    stop_index = 25000

    try:
        opts, args = getopt.getopt(argv, "",
                                   ["directory=", "start_index=", "stop_index="])
    except getopt.GetoptError:
        print '\r\nPlease, use this software this way:'
        print 'Embedding: shrink.py --directory %PATH_TO_DIGITAL_IMAGES% --start_index %start_index% ' \
              '--stop_index %stop_index%'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--directory':
            directory = arg
        if opt == '--start_index':
            start_index = int(arg)
        if opt == '--stop_index':
            stop_index = int(arg)

    os.system("taskset -p 0xff %d" % os.getpid())
    shrink_dir(directory, start_index, stop_index)


if __name__ == "__main__":
    main(sys.argv[1:])
