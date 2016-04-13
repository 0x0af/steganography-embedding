#!/usr/bin/python

# coding: utf-8


"""
Created on Wed Dec 31 10:22:58 2014

@author: d.progonov

Modified, supplemented with pseudorandomness source switch by Anton at 0x0af@ukr.net
"""

import numpy
import pywt
import scipy
from scipy import misc

from py.steganography.embedding_method import *
from py.pseudorandomness_sources.arnold_cat_map import ArnoldCatMapDecoder
from py.pseudorandomness_sources.arnold_cat_map import ArnoldCatMapEncoder
from py.pseudorandomness_sources.logistic_map import LogisticMapEncoder, LogisticMapDecoder


class DeyMethodConfig(EmbeddingMethodConfig):
    DEY_METHOD = 'DeyMethod'

    AUX_G = 'dey_g'
    AUX_WAVELET = 'dey_wavelet'


class DeyEmbedder(Embedder):
    def __init__(self, container, watermark, pseudorandom_encoder=None, g=0.01):
        super(DeyEmbedder, self).__init__(container, watermark, DeyMethodConfig.DEY_METHOD, pseudorandom_encoder)
        # self.aux[DeyMethodConfig.AUX_PAYLOAD] = payload
        self.aux[DeyMethodConfig.AUX_G] = g
        self.aux[DeyMethodConfig.AUX_WAVELET] = 'db1'

    def _embed(self):
        size_c = list(self.container.shape)
        wave_size = numpy.divide(size_c[0:2], 2)

        self.watermark = scipy.misc.imresize(self.watermark, wave_size, interp='bicubic')
        super(DeyEmbedder, self)._embed()

        wave_size = numpy.append(wave_size, self.container.ndim)
        contc_a = numpy.zeros(wave_size, dtype=numpy.single)
        contc_h = numpy.zeros(wave_size, dtype=numpy.single)
        contc_v = numpy.zeros(wave_size, dtype=numpy.single)
        contc_d = numpy.zeros(wave_size, dtype=numpy.single)

        stego = numpy.zeros(size_c, dtype=self.container.dtype)

        for i in range(self.container.ndim):
            contc_a[:, :, i], (contc_h[:, :, i], contc_v[:, :, i], contc_d[:, :, i]) = pywt.dwt2(
                self.container[:, :, i], self.aux[DeyMethodConfig.AUX_WAVELET])

            cas, (chs, cvs, cds) = pywt.dwt2(self.watermark[:, :, i], self.aux[DeyMethodConfig.AUX_WAVELET])

            watermark_coefficients = numpy.zeros(contc_a.shape)

            size = cas.shape[0]

            watermark_coefficients[0:size, 0:size, i] = cas
            watermark_coefficients[size:2 * size, 0:size, i] = chs
            watermark_coefficients[0:size, size:2 * size, i] = cvs
            watermark_coefficients[size:2 * size, size:2 * size, i] = cds

            # print watermark_coefficients.shape

            car = (1 - self.aux[DeyMethodConfig.AUX_G]) * contc_a[:, :, i] + \
                  self.aux[DeyMethodConfig.AUX_G] * watermark_coefficients[:, :, i]
            chr = (1 - self.aux[DeyMethodConfig.AUX_G]) * contc_h[:, :, i] + \
                  self.aux[DeyMethodConfig.AUX_G] * watermark_coefficients[:, :, i]
            cvr = (1 - self.aux[DeyMethodConfig.AUX_G]) * contc_v[:, :, i] + \
                  self.aux[DeyMethodConfig.AUX_G] * watermark_coefficients[:, :, i]
            cdr = (1 - self.aux[DeyMethodConfig.AUX_G]) * contc_d[:, :, i] + \
                  self.aux[DeyMethodConfig.AUX_G] * watermark_coefficients[:, :, i]

            stego[:, :, i] = pywt.idwt2((car, (chr, cvr, cdr)), self.aux[DeyMethodConfig.AUX_WAVELET])

        return stego


class DeyExtractor(Extractor):
    def __init__(self, container, stego, aux, pseudorandomness_source=None):
        super(DeyExtractor, self).__init__(container, stego, aux, DeyMethodConfig.DEY_METHOD, pseudorandomness_source)

    def _extract(self):
        c_a = numpy.zeros((numpy.divide(self.container.shape[0], 2),
                           numpy.divide(self.container.shape[1], 2), self.container.shape[2]),
                          dtype=numpy.single)
        c_h = numpy.copy(c_a)
        c_v = numpy.copy(c_a)
        c_d = numpy.copy(c_a)

        self.watermark = numpy.zeros(self.aux[EmbeddingMethodConfig.AUX_STEGO_SIZE], dtype=self.container.dtype)

        for i in range(self.container.ndim):
            c_a[:, :, i], (c_h[:, :, i], c_v[:, :, i], c_d[:, :, i]) = pywt.dwt2(self.container[:, :, i],
                                                                                 self.aux[DeyMethodConfig.AUX_WAVELET])

            caw, (chw, cvw, cdw) = pywt.dwt2(self.stego[:, :, i], self.aux[DeyMethodConfig.AUX_WAVELET])

            cas = (caw - (1 - self.aux[DeyMethodConfig.AUX_G]) * c_a[:, :, i]) / self.aux[DeyMethodConfig.AUX_G]
            chs = (chw - (1 - self.aux[DeyMethodConfig.AUX_G]) * c_h[:, :, i]) / self.aux[DeyMethodConfig.AUX_G]
            cvs = (cvw - (1 - self.aux[DeyMethodConfig.AUX_G]) * c_v[:, :, i]) / self.aux[DeyMethodConfig.AUX_G]
            cds = (cdw - (1 - self.aux[DeyMethodConfig.AUX_G]) * c_d[:, :, i]) / self.aux[DeyMethodConfig.AUX_G]

            size = cas.shape[0] / 2

            LL = (cas[0:size, 0:size] + chs[0:size, 0:size] + cvs[0:size, 0:size] + cds[0:size, 0:size]) / 4
            LH = (cas[size:2 * size, 0:size] + chs[size:2 * size, 0:size] + cvs[size:2 * size, 0:size] + cds[
                                                                                                         size:2 * size,
                                                                                                         0:size]) / 4
            HL = (cas[0:size, size:2 * size] + chs[0:size, size:2 * size] +
                  cvs[0:size, size:2 * size] + cds[0:size, size:2 * size]) / 4
            HH = (cas[size:2 * size, size:2 * size] + chs[size:2 * size, size:2 * size] +
                  cvs[size:2 * size, size:2 * size] + cds[size:2 * size, size:2 * size]) / 4

            self.watermark[:, :, i] = pywt.idwt2((LL, (LH, HL, HH)), self.aux[DeyMethodConfig.AUX_WAVELET])
        super(DeyExtractor, self)._extract()
        return self.watermark

# print EmbeddingMethodTestSuite.test_method(DeyEmbedder, DeyExtractor, 512, LogisticMapEncoder, LogisticMapDecoder)

# x512
#  ---
# Test passed
# Encode time: 2.026741266737656
# Decode time: 1.9711947158966954
#  ---

# x1024
#  ---
# Test passed
# Encode time: 5.086670263993156
# Decode time: 3.8779849774004456
#  ---

# EmbeddingMethodTestSuite.test_method_subjective_quality(DeyEmbedder, DeyExtractor, LogisticMapEncoder,
#                                                         LogisticMapDecoder)
