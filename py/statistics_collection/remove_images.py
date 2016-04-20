import getopt
import os
import sys

from scipy.io import loadmat, savemat


def remove_images_from_dir(directory, start_index, stop_index):
    filenames = {}

    for i in range(start_index, stop_index + 1, 1):
        filenames[i * 4] = 'im' + str(i) + '.jpg-c.mat'
        filenames[i * 4 + 1] = 'im' + str(i) + '.jpg-am.mat'
        filenames[i * 4 + 2] = 'im' + str(i) + '.jpg-lm.mat'
        filenames[i * 4 + 3] = 'im' + str(i) + '.jpg-he.mat'

    for f in filenames.itervalues():
        try:
            structure = loadmat(directory + '/' + f)
            structure_with_no_image = removekey(structure, 'image')
            savemat(directory + '/' + f, structure_with_no_image)
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
        print 'Embedding: remove_images.py --directory %PATH_TO_DIGITAL_IMAGES% --start_index %start_index% ' \
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
    remove_images_from_dir(directory, start_index, stop_index)


if __name__ == "__main__":
    main(sys.argv[1:])
