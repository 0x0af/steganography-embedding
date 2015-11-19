#!/usr/bin/python

# coding: utf-8

"""
Created on Sat Nov 7 2015

@author: Anton at 0x0af@ukr.net
"""

__all__ = ["EmbeddingMethodConfig", "EmbeddingMethodStack", "Embedder", "Extractor", "EmbeddingMethodTestSuite"]


class EmbeddingMethodConfig(object):
    AUX_METHOD_NAME = 'method_name'
    AUX_STEGO_SIZE = 'stego_size'

    AUX_PSEUDORANDOMNESS_CONFIG = 'pseudorandomness_config'


class EmbeddingMethodStack(object):
    @staticmethod
    def pseudo_rand_mask_create(size_p):
        import numpy
        position = numpy.zeros((size_p[0:2].prod(), 2), dtype=numpy.int32)

        x = numpy.tile(range(size_p[1]), (size_p[0], 1))
        y = numpy.tile(range(size_p[0]), (size_p[1], 1)).T

        for i in range(size_p[0]):
            y[:, i] = numpy.roll(y[:, i], i)

        position[:, 0] = numpy.reshape(y, (y.size, 1)).squeeze()
        position[:, 1] = numpy.reshape(x, (x.size, 1)).squeeze()

        return position

    @staticmethod
    def matrix_2_bitstream(img):
        import numpy
        return numpy.unpackbits(img)

    @staticmethod
    def bitstream_2_matrix(bit_stream, size_i):
        import numpy
        return numpy.reshape(numpy.packbits(bit_stream), (size_i[0], size_i[1]))

    @staticmethod
    def ycbcr_2_rgb(im, init_type):
        import numpy
        im_rec = numpy.zeros(im.shape, dtype=im.dtype)

        y = im[:, :, 0].astype('single')
        cb = im[:, :, 1].astype('single')
        cr = im[:, :, 2].astype('single')

        im_rec[:, :, 0] = y + 0.00000 * (cb - 0.0) + 1.40200 * (cr - 128.0)  # R-channel
        im_rec[:, :, 1] = y + (-0.34414) * (cb - 128.0) + (-0.71414) * (cr - 128.0)  # G-channel
        im_rec[:, :, 2] = y + 1.77200 * (cb - 128.0) + 0.00000 * (cr - 0.0)  # B-channel

        return numpy.round(im_rec).astype(init_type)  # reconstruct for specified data type

    @staticmethod
    def rgb_2_ycbcr(im):
        import numpy
        im_out = numpy.zeros(im.shape, dtype='single')

        r = im[:, :, 0].astype('single')
        g = im[:, :, 1].astype('single')
        b = im[:, :, 2].astype('single')

        im_out[:, :, 0] = 0.29900 * r + 0.58700 * g + 0.11400 * b  # Y-channel
        im_out[:, :, 1] = (-0.16874) * r + (-0.33126) * g + 0.50000 * b + 128.0  # Cb-channel
        im_out[:, :, 2] = 0.50000 * r + (-0.41869) * g + (-0.08131) * b + 128.0  # Cr-channel

        return im_out

    @staticmethod
    def reshape_matrix_to_compliance(matrix):
        import math
        import numpy
        import scipy
        from scipy import misc
        if EmbeddingMethodStack.check_matrix_compliance(matrix):
            return matrix
        block_size = 2 ** int(math.ceil(math.log(numpy.amax(matrix.shape[0:2]), 2)) - 1)
        n = numpy.amax(matrix.shape[0:2]) - numpy.amax(matrix.shape[0:2]) % block_size
        matrix = scipy.misc.imresize(matrix, numpy.array([n, n]), interp='bicubic')
        return matrix.astype('uint8')

    @staticmethod
    def is_power_of_2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    @staticmethod
    def check_compliance_by_channels(image):
        compliant = True
        for i in range(image.ndim):
            compliant &= EmbeddingMethodStack.check_matrix_compliance(image[:, :, i])
        return compliant

    @staticmethod
    def check_matrix_compliance(matrix):
        return matrix.shape[0] == matrix.shape[1] and EmbeddingMethodStack.is_power_of_2(matrix.shape[0])

    @staticmethod
    def reshape_to_compliance_by_channels(image):
        import numpy
        temp = numpy.copy(image)
        if EmbeddingMethodStack.check_compliance_by_channels(image):
            return image
        for i in range(numpy.ndim(image)):
            temp[:, :, i] = EmbeddingMethodStack.reshape_matrix_to_compliance(image[:, :, i])
        return temp

    @staticmethod
    def randomize_matrix(matrix, pseudorandom_image_encoder):
        encoder = pseudorandom_image_encoder(matrix)
        return encoder.encode(False)

    @staticmethod
    def derandomize_matrix(matrix, pseudorandom_image_decoder, pseudorandomness_config):
        decoder = pseudorandom_image_decoder(matrix, pseudorandomness_config)
        return decoder.decode()


class Embedder(object):
    def __init__(self, container, watermark, method_name=None, pseudorandom_encoder=None):
        self.aux = {EmbeddingMethodConfig.AUX_METHOD_NAME: method_name}
        self.container = EmbeddingMethodStack.reshape_to_compliance_by_channels(container)
        self.watermark = EmbeddingMethodStack.reshape_to_compliance_by_channels(watermark)
        self.pseudorandom_encoder = pseudorandom_encoder

    def _embed(self):
        # override to specify the embedding procedure, call to apply pseudorandom_encoder to watermark
        if self.pseudorandom_encoder is not None:
            import numpy
            self.aux[EmbeddingMethodConfig.AUX_PSEUDORANDOMNESS_CONFIG] = {}
            temp = numpy.copy(self.watermark)
            if self.watermark.ndim > 2:
                for i in range(self.watermark.ndim):
                    temp[:, :, i], channel_aux = EmbeddingMethodStack.randomize_matrix(self.watermark[:, :, i],
                                                                                       self.pseudorandom_encoder)
                    self.aux[EmbeddingMethodConfig.AUX_PSEUDORANDOMNESS_CONFIG][i] = channel_aux
                self.watermark = temp
            else:
                temp, channel_aux = EmbeddingMethodStack.randomize_matrix(self.watermark,
                                                                          self.pseudorandom_encoder)
                self.aux[EmbeddingMethodConfig.AUX_PSEUDORANDOMNESS_CONFIG][0] = channel_aux
                self.watermark = temp
        self.aux[EmbeddingMethodConfig.AUX_STEGO_SIZE] = self.watermark.shape

    def embed(self):
        # returns resized container and watermark
        return self._embed(), self.container, self.watermark, self.aux


class Extractor(object):
    def __init__(self, container, stego, aux, method_name=None, pseudorandom_extractor=None):
        self.aux = aux
        self.container = container
        self.stego = stego
        self.pseudorandom_extractor = pseudorandom_extractor
        if self.aux[EmbeddingMethodConfig.AUX_METHOD_NAME] != method_name:
            raise TypeError('The provided method configuration is not compliant with extractor type')
        if not EmbeddingMethodStack.check_compliance_by_channels(self.container):
            raise TypeError('The provided container image is not compliant with the stack')
        if not EmbeddingMethodStack.check_compliance_by_channels(self.stego):
            raise TypeError('The provided stego image is not compliant with the stack')
        self.watermark = None

    def _extract(self):
        # override to specify the extracting procedure
        if self.pseudorandom_extractor is not None and self.watermark is not None:
            import numpy
            temp = numpy.zeros(self.watermark.shape)
            if self.pseudorandom_extractor is not None:
                if self.watermark.ndim > 2:
                    for i in range(self.watermark.ndim):
                        temp[:, :, i] = EmbeddingMethodStack.derandomize_matrix(
                            self.watermark[:, :, i], self.pseudorandom_extractor,
                            self.aux[EmbeddingMethodConfig.AUX_PSEUDORANDOMNESS_CONFIG][i])
                else:
                    temp = EmbeddingMethodStack.derandomize_matrix(
                        self.watermark, self.pseudorandom_extractor,
                        self.aux[EmbeddingMethodConfig.AUX_PSEUDORANDOMNESS_CONFIG][0])
                if self.watermark.ndim == 1:
                    self.watermark = temp
                if self.watermark.ndim == 3:
                    self.watermark = numpy.dstack((temp[:, :, 0], temp[:, :, 1], temp[:, :, 2]))

    def extract(self):
        return self._extract()


class BlindEmbedder(object):
    def __init__(self, container, watermark, method_name=None):
        self.aux = {EmbeddingMethodConfig.AUX_METHOD_NAME: method_name}
        self.container = container
        self.watermark = watermark
        pass

    def _embed(self):
        # TODO: stub
        pass

    def embed(self):
        # TODO: stub
        return self._embed(), self.container, self.watermark, self.aux


class BlindExtractor(object):
    def __init__(self, stego, aux, method_name=None):
        self.stego = stego
        self.aux = aux
        if self.aux[EmbeddingMethodConfig.AUX_METHOD_NAME] != method_name:
            raise TypeError('The provided method configuration is not compliant with extractor type')
        if not EmbeddingMethodStack.check_matrix_compliance(self.stego):
            raise TypeError('The provided stego image is not compliant with the stack')

    def _extract(self):
        # TODO: stub
        pass

    def extract(self):
        # TODO: stub
        return self._extract()


class DoubleEmbedder(object):
    def __init__(self, container, primary_watermark, secondary_watermark, aux):
        # TODO: stub
        pass

    def _embed(self):
        # TODO: stub
        pass

    def embed(self):
        return self._embed(), self.container, self.primary_watermark, self.second_watermark, self.aux


class DoubleExtractor(object):
    def __init__(self, container, stego, aux):
        # TODO: stub
        pass

    def _extract_primary(self):
        # TODO: stub
        pass

    def _extract_secondary(self):
        # TODO: stub
        pass

    def extract(self):
        # TODO: stub
        return self._extract_primary(), self._extract_secondary()


class EmbeddingMethodTestSuite(object):
    # OBEY THE TESTING GOAT

    @staticmethod
    def test_method_subjective_quality(method_embedder, method_extractor, pseudorandom_image_encoder=None,
                                       pseudorandom_image_decoder=None, color_mode='RGB'):
        import numpy
        import scipy
        from scipy import misc
        lena = numpy.asarray(scipy.misc.lena())
        lena_rgb = numpy.dstack((lena, lena, lena))

        emb = method_embedder(lena_rgb, lena_rgb, pseudorandom_image_encoder)
        encoded_picture, container, watermark, aux = emb.embed()

        ext = method_extractor(container, encoded_picture, aux, pseudorandom_image_decoder)
        decoded_picture = ext.extract()

        if color_mode == 'RGB':
            EmbeddingMethodTestSuite.display_comparison_per_channel_rgb(lena_rgb, encoded_picture, decoded_picture)
        elif color_mode == 'YCbCr':
            EmbeddingMethodTestSuite.display_comparison_per_channel_ycbcr(lena_rgb, encoded_picture, decoded_picture)

    @staticmethod
    def test_method(method_embedder, method_extractor, size, pseudorandom_image_encoder=None,
                    pseudorandom_image_decoder=None):
        import numpy
        from numpy.random import mtrand
        import time
        test_picture = mtrand.rand(size, size) * 255
        test_picture_rgb = numpy.dstack((test_picture, test_picture, test_picture))

        encode_start = time.clock()
        emb = method_embedder(test_picture_rgb, test_picture_rgb, pseudorandom_image_encoder)
        encoded_picture, container, watermark, aux = emb.embed()

        encode_end = time.clock()

        decode_start = time.clock()

        ext = method_extractor(container, encoded_picture, aux, pseudorandom_image_decoder)
        decoded_picture = ext.extract()

        decode_end = time.clock()

        return '\n --- \nTest passed' + '\nEncode time: ' + repr(
            encode_end - encode_start) + '\nDecode time: ' + repr(
            decode_end - decode_start) + '\n --- \n'

    @staticmethod
    def display_comparison_per_channel_rgb(test_picture, encoded_picture, decoded_picture):
        from matplotlib import cm
        from matplotlib import pylab
        f = pylab.figure()

        for i in range(0, 3):
            if i == 0:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('R')
            elif i == 1:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('G')
            elif i == 2:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('B')

            pylab.imshow(test_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Container')

            if i == 0:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('R')
            elif i == 1:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('G')
            elif i == 2:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('B')

            pylab.imshow(encoded_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Stego')

            if decoded_picture.ndim > 2:
                if i == 0:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('R')
                elif i == 1:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('G')
                elif i == 2:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('B')
                pylab.imshow(decoded_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')
            else:
                if i == 0:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('R')
                    pylab.imshow(decoded_picture, cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Extracted Watermark')
        pylab.show()

    @staticmethod
    def display_comparison_per_channel_ycbcr(test_picture, encoded_picture, decoded_picture):
        from matplotlib import cm
        from matplotlib import pylab
        f = pylab.figure()

        for i in range(0, 3):
            if i == 0:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('Y')
            elif i == 1:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('Cb')
            elif i == 2:
                f.add_subplot(3, 3, i * 3 + 1).set_ylabel('Cr')

            pylab.imshow(test_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Container')

            if i == 0:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('Y')
            elif i == 1:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('Cb')
            elif i == 2:
                f.add_subplot(3, 3, i * 3 + 2).set_ylabel('Cr')

            pylab.imshow(encoded_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Stego')

            if decoded_picture.ndim > 2:
                if i == 0:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('Y')
                elif i == 1:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('Cb')
                elif i == 2:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('Cr')
                pylab.imshow(decoded_picture[:, :, i], cmap=cm.autumn, interpolation='nearest')
            else:
                if i == 0:
                    f.add_subplot(3, 3, i * 3 + 3).set_ylabel('Y')
                    pylab.imshow(decoded_picture, cmap=cm.autumn, interpolation='nearest')

            if i == 0:
                pylab.title('Extracted Watermark')
        pylab.show()
