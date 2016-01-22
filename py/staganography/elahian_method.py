#!/usr/bin/python

# coding: utf-8


"""
Created on Mon Feb 02 08:26:14 2015

@author: d.progonov

Modified, supplemented with pseudorandomness source switch by Anton at 0x0af@ukr.net

"""
import numpy
import pywt
import scipy
import skimage
from skimage import color
from scipy import misc
from py.staganography.embedding_method import *
from py.pseudorandomness_sources.arnold_cat_map import ArnoldCatMapEncoder, ArnoldCatMapDecoder


class ElahianMethodConfig(EmbeddingMethodConfig):
    ELAHIAN_METHOD = 'ElahianMethod'

    AUX_G = 'elahian_g'
    AUX_WAVELET = 'elahian_wavelet'
    AUX_POSITION = 'elahian_position'


class ElahianEmbedder(Embedder):
    def __init__(self, container, watermark, pseudorandom_encoder=None, g=4):
        super(ElahianEmbedder, self).__init__(container, watermark, ElahianMethodConfig.ELAHIAN_METHOD,
                                              pseudorandom_encoder)
        self.aux[ElahianMethodConfig.AUX_G] = g
        self.aux[ElahianMethodConfig.AUX_WAVELET] = 'db1'

    def _embed(self):
        container = EmbeddingMethodStack.rgb_2_ycbcr(self.container)
        stego = numpy.copy(container)
        self.watermark = skimage.color.rgb2gray(self.watermark)
        self.watermark = scipy.misc.imresize(self.watermark, (numpy.divide(container.shape[0], 64),
                                                              numpy.divide(container.shape[1], 64)), interp='bicubic')
        super(ElahianEmbedder, self)._embed()

        watermark_bitstream = EmbeddingMethodStack.matrix_2_bitstream(self.watermark.astype(numpy.uint8))

        position = EmbeddingMethodStack.pseudo_rand_mask_create(numpy.asarray((numpy.divide(container.shape[0], 8),
                                                                               numpy.divide(container.shape[1], 8))))
        self.aux[ElahianMethodConfig.AUX_POSITION] = position

        c1a, (c1h, c1v, c1d) = pywt.dwt2(container[:, :, 0], self.aux[ElahianMethodConfig.AUX_WAVELET])
        c2a, (c2h, c2v, c2d) = pywt.dwt2(c1a, self.aux[ElahianMethodConfig.AUX_WAVELET])
        c3a, (c3h, c3v, c3d) = pywt.dwt2(c2a, self.aux[ElahianMethodConfig.AUX_WAVELET])

        x = position[:, 1]
        y = position[:, 0]
        c3ae = numpy.copy(c3a)
        # print c3ae.shape, watermark_bitstream.shape, position.shape
        for k in range(watermark_bitstream.size - 1):
            c3ae[y[k], x[k]] = c3a[y[k], x[k]] + self.aux[ElahianMethodConfig.AUX_G] * watermark_bitstream[k]

        c2ae = pywt.idwt2((c3ae, (c3h, c3v, c3d)), self.aux[ElahianMethodConfig.AUX_WAVELET])
        c1ae = pywt.idwt2((c2ae, (c2h, c2v, c2d)), self.aux[ElahianMethodConfig.AUX_WAVELET])
        stego[:, :, 0] = pywt.idwt2((c1ae, (c1h, c1v, c1d)), self.aux[ElahianMethodConfig.AUX_WAVELET])

        stego = EmbeddingMethodStack.ycbcr_2_rgb(stego, self.container.dtype)

        # print stego.shape, self.watermark.shape, self.container.shape

        return stego


class ElahianExtractor(Extractor):
    def __init__(self, container, stego, aux, pseudorandomness_source=None):
        super(ElahianExtractor, self).__init__(container, stego, aux, ElahianMethodConfig.ELAHIAN_METHOD,
                                               pseudorandomness_source)

    def _extract(self):
        container = EmbeddingMethodStack.rgb_2_ycbcr(self.container)
        stego = EmbeddingMethodStack.rgb_2_ycbcr(self.stego)

        c1a, (c1h, c1v, c1d) = pywt.dwt2(container[:, :, 0], self.aux[ElahianMethodConfig.AUX_WAVELET])
        c2a, (c2h, c2v, c2d) = pywt.dwt2(c1a, self.aux[ElahianMethodConfig.AUX_WAVELET])
        c3a, (c3h, c3v, c3d) = pywt.dwt2(c2a, self.aux[ElahianMethodConfig.AUX_WAVELET])

        w1a, (w1h, w1v, w1d) = pywt.dwt2(stego[:, :, 0], self.aux[ElahianMethodConfig.AUX_WAVELET])
        w2a, (w2h, w2v, w2d) = pywt.dwt2(w1a, self.aux[ElahianMethodConfig.AUX_WAVELET])
        w3a, (w3h, w3v, w3d) = pywt.dwt2(w2a, self.aux[ElahianMethodConfig.AUX_WAVELET])

        x = self.aux[ElahianMethodConfig.AUX_POSITION][:, 1]  # coordinates for stegodata embedding (abscissa)
        y = self.aux[ElahianMethodConfig.AUX_POSITION][:, 0]  # coordinates for stegodata embedding (ordinate)

        bit_ammount = self.aux[ElahianMethodConfig.AUX_STEGO_SIZE][0] * \
                      self.aux[ElahianMethodConfig.AUX_STEGO_SIZE][1] * 8
        watermark_bitstream = numpy.zeros((1, bit_ammount), dtype=numpy.uint8)
        for k in range(bit_ammount):
            if w3a[y[k], x[k]] > c3a[y[k], x[k]]:
                watermark_bitstream[0][k] = 1
                w3a[y[k], x[k]] = w3a[y[k], x[k]] - self.aux[ElahianMethodConfig.AUX_G]
            else:
                watermark_bitstream[0][k] = 0

        # print watermark_bitstream.size, self.aux[ElahianMethodConfig.AUX_STEGO_SIZE]

        self.watermark = EmbeddingMethodStack.bitstream_2_matrix(
            watermark_bitstream, self.aux[ElahianMethodConfig.AUX_STEGO_SIZE])

        super(ElahianExtractor, self)._extract()

        return self.watermark


# print EmbeddingMethodTestSuite.test_method(ElahianEmbedder, ElahianExtractor, 256, ArnoldCatMapEncoder,
#                                            ArnoldCatMapDecoder)

# EmbeddingMethodTestSuite.test_method_subjective_quality(ElahianEmbedder, ElahianExtractor, ArnoldCatMapEncoder,
#                                                         ArnoldCatMapDecoder, 'YCbCr')
