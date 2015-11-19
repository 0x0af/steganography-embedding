#!/usr/bin/python

# coding: utf-8

"""
Created on Tue Oct 20 2015

@author: Anton at 0x0af@ukr.net
"""

import numpy
from numpy import *

from pseudorandomness_source import *

__all__ = ["ArnoldCatMapEncoder", "ArnoldCatMapDecoder"]


class ArnoldCatMapStack(object):
    @staticmethod
    def arnold_cat_map(square_matrix, encoding_step):
        n = square_matrix.shape[0]
        x, y = meshgrid(range(n), range(n))
        xmap = (2 * x + y) % n
        ymap = (x + y) % n
        for i in range(encoding_step):
            square_matrix = square_matrix[xmap, ymap]
        return square_matrix

    @staticmethod
    def blockshaped(arr, nrows, ncols):
        h, w = arr.shape
        return (arr.reshape(h / nrows, nrows, -1, ncols)
                .swapaxes(1, 2)
                .reshape(-1, nrows, ncols))

    @staticmethod
    def unblockshaped(arr, h, w):
        n, nrows, ncols = arr.shape
        return (arr.reshape(h / nrows, -1, nrows, ncols)
                .swapaxes(1, 2)
                .reshape(h, w))


class ArnoldCatMapConfig(PseudorandomnessSourceConfig):
    ARNOLD_CAT_MAP = 'ArnoldCatMap'

    AUX_BLOCK_SIZE = 'arnold_block_size'
    AUX_ENCODING_STEP = 'arnold_encoding_step'


class ArnoldCatMapEncoder(PseudorandomImageEncoder):
    def __init__(self, square_matrix):
        super(ArnoldCatMapEncoder, self).__init__(square_matrix, ArnoldCatMapConfig.ARNOLD_CAT_MAP)

        self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE] = 2 ** int(
            math.ceil(math.log(numpy.amax(square_matrix.shape[0:2]), 2)))
        self.aux[ArnoldCatMapConfig.AUX_ENCODING_STEP] = random.randint(1, self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE])

    def _encode(self):
        square_matrix_blocks = ArnoldCatMapStack.blockshaped(self.square_matrix,
                                                             self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE],
                                                             self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE])
        temp = numpy.copy(square_matrix_blocks)
        for idx, value in enumerate(square_matrix_blocks):
            temp[idx] = ArnoldCatMapStack.arnold_cat_map(value, self.aux[ArnoldCatMapConfig.AUX_ENCODING_STEP])
        return ArnoldCatMapStack.unblockshaped(temp, self.aux[ArnoldCatMapConfig.AUX_SQUARE_MATRIX_SHAPE][0],
                                               self.aux[ArnoldCatMapConfig.AUX_SQUARE_MATRIX_SHAPE][1])


class ArnoldCatMapDecoder(PseudorandomImageDecoder):
    def __init__(self, square_matrix, aux):
        super(ArnoldCatMapDecoder, self).__init__(square_matrix, aux, ArnoldCatMapConfig.ARNOLD_CAT_MAP)

    def _decode(self):
        square_matrix_blocks = ArnoldCatMapStack.blockshaped(self.square_matrix,
                                                             self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE],
                                                             self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE])
        temp = numpy.copy(square_matrix_blocks)
        for idx, value in enumerate(square_matrix_blocks):
            temp[idx] = ArnoldCatMapStack.arnold_cat_map(value, self.aux[ArnoldCatMapConfig.AUX_BLOCK_SIZE] - self.aux[
                ArnoldCatMapConfig.AUX_ENCODING_STEP])
        return ArnoldCatMapStack.unblockshaped(temp, self.aux[ArnoldCatMapConfig.AUX_SQUARE_MATRIX_SHAPE][0],
                                               self.aux[ArnoldCatMapConfig.AUX_SQUARE_MATRIX_SHAPE][1])


# print PseudorandomnessSourceTestSuite.test_source(ArnoldCatMapEncoder, ArnoldCatMapDecoder, 1024)

# x128
#  ---
# Test passed
# Encode time: 0.06183571707505897
# Decode time: 0.024902937155526918
#  ---

# x512
#  ---
# Test passed
# Encode time: 0.9038658949512275
# Decode time: 2.9583216215051733
#  ---

# x1024
#  ---
# Test passed
# Encode time: 19.910280899222197
# Decode time: 7.764746473970334
#  ---

# PseudorandomnessSourceTestSuite.test_source_subjective_quality(ArnoldCatMapEncoder, ArnoldCatMapDecoder)
