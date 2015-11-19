#!/usr/bin/python

# coding: utf-8

"""
Created on Tue Oct 20 2015

@author: Anton at 0x0af@ukr.net

"""
import math
import numpy
import scipy
from scipy import misc

__all__ = ["PseudorandomnessSourceConfig", "PseudorandomImageEncoder", "PseudorandomImageDecoder",
           "PseudorandomnessSourceTestSuite"]


class PseudorandomnessSourceConfig(object):
    AUX_SQUARE_MATRIX_SHAPE = 'square_matrix_shape'
    AUX_ENCODER_TYPE = 'encoder_type'


class PseudorandomnessSourceStack(object):
    @staticmethod
    def reshape_matrix_to_compliance(matrix):
        import math
        import numpy
        import scipy
        from scipy import misc
        if PseudorandomnessSourceStack.check_matrix_compliance(matrix):
            return matrix
        block_size = 2 ** int(math.ceil(math.log(numpy.amax(matrix.shape[0:2]), 2)) - 1)
        n = numpy.amax(matrix.shape[0:2]) - numpy.amax(matrix.shape[0:2]) % block_size
        matrix = scipy.misc.imresize(matrix, numpy.array([n, n]), interp='bicubic')
        return matrix.astype('uint8')

    @staticmethod
    def is_power_of_2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    @staticmethod
    def check_matrix_compliance(matrix):
        return matrix.shape[0] == matrix.shape[1] and PseudorandomnessSourceStack.is_power_of_2(matrix.shape[0])


class PseudorandomImageEncoder(object):
    def __init__(self, square_matrix, encoder_type=None):
        self.aux = {PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE: square_matrix.shape,
                    PseudorandomnessSourceConfig.AUX_ENCODER_TYPE: encoder_type}
        PseudorandomnessSourceStack.reshape_matrix_to_compliance(square_matrix)
        self.square_matrix = square_matrix.astype('uint8')

    # TODO: add convenience methods

    # def __repr__(self):
    #     return '\n --- \nPseudorandomness decoder details' + '\nEncoder type: ' + self.aux[
    #         PseudorandomnessSourceConfig.AUX_ENCODER_TYPE] + '\nSquare matrix shape: ' + repr(
    #         self.aux[PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE]) + '\n --- \n'

    def _encode(self):
        # Do the actual encoding here, return encoded image matrix
        return self.square_matrix

    def encode(self, unpack_to_bitstream):
        encoder_output = self._encode()
        if unpack_to_bitstream:
            return numpy.unpackbits(encoder_output), self.aux
        return encoder_output, self.aux


class PseudorandomImageDecoder(object):
    def __init__(self, square_matrix, aux, encoder_type=None):
        self.aux = aux
        self.square_matrix = square_matrix.astype('uint8')
        if self.aux[PseudorandomnessSourceConfig.AUX_ENCODER_TYPE] != encoder_type:
            raise TypeError('The provided decoder configuration does not comply with decoder type')

    @classmethod
    def from_bitstream(cls, bitstream, aux):
        return cls(numpy.reshape(numpy.packbits(bitstream), aux[PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE]),
                   aux)

    def __repr__(self):
        return '\n --- \nPseudorandomness decoder details' + '\nEncoder type: ' + self.aux[
            PseudorandomnessSourceConfig.AUX_ENCODER_TYPE] + '\nSquare matrix shape: ' + repr(
            self.aux[PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE]) + '\n --- \n'

    def _decode(self):
        # Do the actual decoding here, return decoded image matrix
        return self.square_matrix

    def decode(self):
        return self._decode()


class PseudorandomnessSourceTestSuite(object):
    # OBEY THE TESTING GOAT

    @staticmethod
    def compare(picture_1, picture_2):
        err = numpy.sum((picture_1.astype("float") - picture_2.astype("float")))
        err /= float(picture_1.shape[0] * picture_2.shape[1])
        return err < 50

    @staticmethod
    def test_source_subjective_quality(source_encoder, source_decoder):
        lena = numpy.asarray(scipy.misc.lena())
        enc = source_encoder(lena)
        encoded_picture, aux = enc.encode(False)
        dec = source_decoder(encoded_picture, aux)
        decoded_picture = dec.decode()
        from matplotlib import cm
        from matplotlib import pylab
        f = pylab.figure()
        f.add_subplot(1, 3, 1)
        pylab.imshow(lena, cmap=cm.autumn, interpolation='nearest')
        pylab.title('Pre-encoded Image')
        f.add_subplot(1, 3, 2)
        pylab.imshow(encoded_picture, cmap=cm.autumn, interpolation='nearest')
        pylab.title('Encoded Image')
        f.add_subplot(1, 3, 3)
        pylab.imshow(decoded_picture, cmap=cm.autumn, interpolation='nearest')
        pylab.title('Decoded Image')
        pylab.show()

    @staticmethod
    def test_source(source_encoder, source_decoder, size):
        from numpy.random import mtrand
        import scipy
        from scipy import misc
        import time
        test_picture = mtrand.rand(size, size) * 255

        encode_start = time.clock()

        enc = source_encoder(test_picture)
        encoded_picture, aux = enc.encode(False)

        encode_end = time.clock()

        decode_start = time.clock()

        dec = source_decoder(encoded_picture, aux)
        decoded_picture = dec.decode()

        decode_end = time.clock()

        if not PseudorandomnessSourceTestSuite.compare(
                scipy.misc.imresize(test_picture, numpy.array(
                    [aux[PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE][0],
                     aux[PseudorandomnessSourceConfig.AUX_SQUARE_MATRIX_SHAPE][1]]), interp='bicubic'),
                decoded_picture):
            return 'Test failed'

        return '\n --- \nTest passed' + '\nEncode time: ' + repr(encode_end - encode_start) + '\nDecode time: ' + repr(
            decode_end - decode_start) + '\n --- \n'
