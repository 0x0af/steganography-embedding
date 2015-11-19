#!/usr/bin/python

# coding: utf-8

"""
Created on Tue Oct 20 2015

@author: Anton at 0x0af@ukr.net
"""

import numpy
from numpy import *
from pseudorandomness_source import *

__all__ = ["LogisticMapEncoder", "LogisticMapDecoder"]


class LogisticMap(object):
    def __init__(self, x, r):
        self.x = x
        self.r = r
        self.iteration = 0

    def __iteration(self):
        temp = self.x * self.r * (1 - self.x)
        self.x = temp
        self.iteration += 1

    def get_nth_iteration(self, n):
        if n == self.iteration:
            return self.x
        elif n < self.iteration:
            raise ValueError('The requested iteration has already been queried')
        else:
            for _ in range(n - self.iteration):
                self.__iteration()
            return self.x


class LogisticMapConfig(PseudorandomnessSourceConfig):
    LOGISTIC_MAP = 'LogisticMap'

    AUX_ENCODING_PARAMETER_X = 'logistic_x'
    AUX_ENCODING_PARAMETER_R = 'logistic_r'


class LogisticMapEncoder(PseudorandomImageEncoder):
    def __init__(self, square_matrix):
        super(LogisticMapEncoder, self).__init__(square_matrix, LogisticMapConfig.LOGISTIC_MAP)
        self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_X] = random.uniform(0, 1)
        self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_R] = 4  # random.uniform(3.75, 3.8)
        self.logistic_map = LogisticMap(self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_X],
                                        self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_R])

    def _encode(self):
        # print self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_X]
        pixel_mapping_table = numpy.zeros(256)
        i = 0
        n = i
        while i < 255:
            temporary_level = int(round(255 * self.logistic_map.get_nth_iteration(n)))
            n += 1
            if temporary_level in pixel_mapping_table:
                while temporary_level in pixel_mapping_table:
                    temporary_level = int(round(255 * self.logistic_map.get_nth_iteration(n)))
                    n += 1
            pixel_mapping_table[i] = temporary_level
            i += 1

        # print pixel_mapping_table

        # from matplotlib import pyplot as plt
        #
        # plt.figure()
        # plt.plot(pixel_mapping_table)
        # plt.show()

        # print numpy.unique(pixel_mapping_table).shape

        temp = numpy.copy(self.square_matrix)
        for (x, y), element in numpy.ndenumerate(temp):
            if 255 >= pixel_mapping_table[element] >= 0:
                temp[x][y] = pixel_mapping_table[element]
            elif pixel_mapping_table[element] > 255:
                temp[x][y] = 255
            elif pixel_mapping_table[element] < 0:
                temp[x][y] = 0

                # print 'temp[' + repr(x) + ']' + '[' + repr(y) + '] == ' + repr(temp[x][y]) + ', was ' + repr(element)
        return temp


class LogisticMapDecoder(PseudorandomImageDecoder):
    def __init__(self, square_matrix, aux):
        super(LogisticMapDecoder, self).__init__(square_matrix, aux, LogisticMapConfig.LOGISTIC_MAP)
        self.logistic_map = LogisticMap(self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_X],
                                        self.aux[LogisticMapConfig.AUX_ENCODING_PARAMETER_R])

    def _decode(self):
        pixel_mapping_table = numpy.zeros(256)
        i = 0
        n = i
        while i < 255:
            temporary_level = int(round(255 * self.logistic_map.get_nth_iteration(n)))
            n += 1
            if temporary_level in pixel_mapping_table:
                while temporary_level in pixel_mapping_table:
                    temporary_level = int(round(255 * self.logistic_map.get_nth_iteration(n)))
                    n += 1
            pixel_mapping_table[i] = temporary_level
            i += 1

        # print pixel_mapping_table

        temp = numpy.copy(self.square_matrix)
        for (x, y), element in numpy.ndenumerate(temp):
            itemindex = numpy.where(pixel_mapping_table == element)
            temp[x][y] = itemindex[0][0]
            # print 'temp[' + repr(x) + ']' + '[' + repr(y) + '] == ' + repr(temp[x][y]) + ', was ' + repr(element)
        return temp

# print PseudorandomnessSourceTestSuite.test_source(LogisticMapEncoder, LogisticMapDecoder, 1024)

# x128
#   ---
# Test passed
# Encode time: 0.32308589804176896
# Decode time: 0.5708483251458261
#  ---

# x512
#  ---
# Test passed
# Encode time: 4.027947177754966
# Decode time: 8.726630725071418
#  ---

# x1024
#  ---
# Test passed
# Encode time: 17.667336890993482
# Decode time: 34.027890222726484
#  ---

# PseudorandomnessSourceTestSuite.test_source_subjective_quality(LogisticMapEncoder, LogisticMapDecoder)
