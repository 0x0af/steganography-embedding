import getopt
import os
import sys

import numpy
from PIL import Image

from py.multifractal_analysis.mfdfa_di_analysis import analyze_image
from py.pseudorandomness_sources.arnold_cat_map import ArnoldCatMapEncoder
from py.pseudorandomness_sources.huffman_encoding import HuffmanEncoder
from py.pseudorandomness_sources.logistic_map import LogisticMapEncoder
from py.steganography.dey_method import DeyEmbedder

OPTIMAL_SKIP = 16
STEGO_NAME = 'stego.png'
G = 0.01


def analyze_dir(directory, start_index, stop_index):
    watermark = numpy.asarray(Image.open(directory + '/' + STEGO_NAME))
    filenames = {}

    for i in range(start_index, stop_index + 1, 1):
        filenames[i] = 'im' + str(i) + '.jpg'

    for f in filenames.itervalues():
        try:
            c_image = Image.open(directory + '/' + f)
            c_image.resize((512, 512), Image.ANTIALIAS)
            container = numpy.asarray(c_image)
            c_name = f + '-c'
            am_name = f + '-am'
            he_name = f + '-he'
            lm_name = f + '-lm'

            arnold_embedder = DeyEmbedder(container, watermark, ArnoldCatMapEncoder, G)
            arnold_steganogram, c, w, aux = arnold_embedder.embed()
            huffman_embedder = DeyEmbedder(container, watermark, HuffmanEncoder, G)
            huffman_steganogram, c, w, aux = huffman_embedder.embed()
            logistic_embedder = DeyEmbedder(container, watermark, LogisticMapEncoder, G)
            logistic_steganogram, c, w, aux = logistic_embedder.embed()
            analyze_image(c_name, container, OPTIMAL_SKIP)
            analyze_image(am_name, arnold_steganogram, OPTIMAL_SKIP)
            analyze_image(he_name, huffman_steganogram, OPTIMAL_SKIP)
            analyze_image(lm_name, logistic_steganogram, OPTIMAL_SKIP)
        except Exception, e:
            print e
            break


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

    os.system("taskset -p 0xff %d" % os.getpid())
    analyze_dir(directory, start_index, stop_index)


if __name__ == "__main__":
    main(sys.argv[1:])
