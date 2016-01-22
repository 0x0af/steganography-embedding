# coding: utf-8


"""
Created on Thu Jul 23 2015

@author: Anton at 0x0af@ukr.net
"""

import sys
import getopt

import numpy
from PIL import Image
from scipy.linalg import hadamard, svd


def bhatnagar_embed(grayscale_container_path, grayscale_watermark_path, watermarked_image_path, alpha):
    """    
    Bhatnagar embedding method implementation. 
    
    Outputs the resulting watermarked image to watermarked_image_path
    
    23-July-2015
    """

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    grayscale_watermark_2darray = numpy.asarray(Image.open(grayscale_watermark_path).convert("L"))

    # Resize container

    while not (is_power2(grayscale_container_2darray.shape[0])):
        grayscale_container_2darray = numpy.r_[
            grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[1])[numpy.newaxis]]

    while not (is_power2(grayscale_container_2darray.shape[1])):
        grayscale_container_2darray = numpy.c_[
            grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[0])]

    while grayscale_container_2darray.shape[0] != grayscale_container_2darray.shape[1]:
        if grayscale_container_2darray.shape[0] > grayscale_container_2darray.shape[1]:
            grayscale_container_2darray = numpy.c_[
                grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[0])]
        elif grayscale_container_2darray.shape[0] < grayscale_container_2darray.shape[1]:
            grayscale_container_2darray = numpy.r_[
                grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[1])[numpy.newaxis]]

    # Coarsest level watermark embed

    coarsest_level_watermarked_image = coarsest_level_embed(grayscale_container_2darray, grayscale_watermark_2darray,
                                                            alpha)

    # Finest level watermark embed

    finest_level_watermarked_image = finest_level_embed(coarsest_level_watermarked_image, grayscale_watermark_2darray,
                                                        alpha)

    finest_level_watermarked_image[finest_level_watermarked_image > 255] = 255
    finest_level_watermarked_image[finest_level_watermarked_image < 0] = 0

    watermarked_image = Image.fromarray(numpy.uint8(finest_level_watermarked_image))
    # watermarked_image.show()

    # Write image to file

    watermarked_image.save(watermarked_image_path)

    return


def finest_level_embed(grayscale_container_2darray, grayscale_watermark_2darray, alpha):
    # Get a 2-level 2D MR-WHT

    transformed_grayscale_stego = multiresolutional_walsh_hadamard_transform(
        multiresolutional_walsh_hadamard_transform(grayscale_container_2darray))

    # TODO: Select _actually_ hh sub-band

    hh = numpy.copy(transformed_grayscale_stego)

    # print hh.shape, transformed_grayscale_stego.shape

    # for row in range(0, int(transformed_grayscale_stego.shape[0] / 2 - 1)):
    #     for cell in range(0, int(transformed_grayscale_stego.shape[0] / 2 - 1)):
    #         hh[row][cell] = transformed_grayscale_stego[row][cell]

    # Perform SVD on hh

    hhu, hhs, hhv = numpy.linalg.svd(hh, full_matrices=True)

    # Perform SVD on watermark

    wu, ws, wv = numpy.linalg.svd(grayscale_watermark_2darray, full_matrices=True)

    # Modify singular values of hh

    middle_sv_offset = 32

    for i in range(0, ws.shape[0]):
        hhs[i + middle_sv_offset] += (float(alpha) * float(ws[i])) / float(numpy.amax(hhs))

    # Perform ISVD on hh

    hh = numpy.dot(hhu, numpy.dot(numpy.diag(hhs), hhv))

    # Get an inverse 2-level 2D MR-WHT

    inverse_transformed_grayscale_stego = inverse_multiresolutional_walsh_hadamard_transform(
        inverse_multiresolutional_walsh_hadamard_transform(hh))

    return inverse_transformed_grayscale_stego


def coarsest_level_embed(grayscale_container_2darray, grayscale_watermark_2darray, alpha):
    # Get a 1-level 2D MR-WHT

    transformed_grayscale_stego = multiresolutional_walsh_hadamard_transform(grayscale_container_2darray)

    # TODO: Select _actually_ hh sub-band

    hh = numpy.copy(transformed_grayscale_stego)

    # print hh.shape, transformed_grayscale_stego.shape

    # for row in range(0, transformed_grayscale_stego.shape[0] - 1)):
    #     for cell in range(0, transformed_grayscale_stego.shape[0] -1)):
    #         hh[row][cell] = transformed_grayscale_stego[row][cell]

    # Perform SVD on hh

    hhu, hhs, hhv = svd(hh, full_matrices=True)

    # Perform SVD on watermark

    wu, ws, wv = svd(grayscale_watermark_2darray, full_matrices=True)

    # Modify singular values of hh

    middle_sv_offset = 63

    for i in range(0, ws.shape[0]):
        hhs[i + middle_sv_offset] += (float(alpha) * float(ws[i])) / float(numpy.amax(hhs))

    # Perform ISVD on hh

    hh = numpy.dot(hhu, numpy.dot(numpy.diag(hhs), hhv))

    # Get an inverse 1-level 2D MR-WHT

    inverse_transformed_grayscale_stego = inverse_multiresolutional_walsh_hadamard_transform(hh)

    return inverse_transformed_grayscale_stego


def bhatnagar_extract(grayscale_stego_path, grayscale_container_path, grayscale_watermark_path,
                      coarsest_extracted_watermark_path,
                      finest_extracted_watermark_path, watermark_size, alpha):
    """
    Bhatnagar extracting method implementation.
    
    Outputs the extracted watermark to ExtractedWatermarkPath
    
    23-July-2015
    """

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    grayscale_stego_2darray = numpy.asarray(Image.open(grayscale_stego_path).convert("L"))
    grayscale_watermark_2darray = numpy.asarray(Image.open(grayscale_watermark_path).convert("L"))
    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    assert grayscale_stego_2darray.shape[0] == grayscale_stego_2darray.shape[1], 'Grayscale Stego is not square!'

    # Resize container

    while not (is_power2(grayscale_container_2darray.shape[0])):
        grayscale_container_2darray = numpy.r_[
            grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[1])[numpy.newaxis]]

    while not (is_power2(grayscale_container_2darray.shape[1])):
        grayscale_container_2darray = numpy.c_[
            grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[0])]

    while grayscale_container_2darray.shape[0] != grayscale_container_2darray.shape[1]:
        if grayscale_container_2darray.shape[0] > grayscale_container_2darray.shape[1]:
            grayscale_container_2darray = numpy.c_[
                grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[0])]
        elif grayscale_container_2darray.shape[0] < grayscale_container_2darray.shape[1]:
            grayscale_container_2darray = numpy.r_[
                grayscale_container_2darray, numpy.zeros(grayscale_container_2darray.shape[1])[numpy.newaxis]]

    assert grayscale_stego_2darray.shape[0] == grayscale_container_2darray.shape[
        1], 'Grayscale Container and Grayscale Stego sizes do not comply!'

    # Extract watermark from the coarsest level

    coarsest_extracted_watermark = coarsest_level_extract(grayscale_stego_2darray, grayscale_container_2darray,
                                                          grayscale_watermark_2darray,
                                                          watermark_size,
                                                          alpha)

    coarsest_extracted_watermark[coarsest_extracted_watermark > 255] = 255
    coarsest_extracted_watermark[coarsest_extracted_watermark < 0] = 0

    # Write image to file

    Image.fromarray(numpy.uint8(coarsest_extracted_watermark)).save(coarsest_extracted_watermark_path)

    # Extract watermark from the finest level

    finest_extracted_watermark = finest_level_extract(grayscale_stego_2darray, grayscale_container_2darray,
                                                      grayscale_watermark_2darray,
                                                      watermark_size,
                                                      alpha)

    finest_extracted_watermark[finest_extracted_watermark > 255] = 255
    finest_extracted_watermark[finest_extracted_watermark < 0] = 0

    # Write image to file

    Image.fromarray(numpy.uint8(finest_extracted_watermark)).save(finest_extracted_watermark_path)

    return


def finest_level_extract(grayscale_stego_2darray, grayscale_container_2darray, grayscale_watermark_2darray,
                         watermark_size, alpha):
    # Get a 2-level 2D MR-WHT of Stego and Container

    transformed_grayscale_stego = multiresolutional_walsh_hadamard_transform(
        multiresolutional_walsh_hadamard_transform(grayscale_stego_2darray))
    transformed_grayscale_container = multiresolutional_walsh_hadamard_transform(
        multiresolutional_walsh_hadamard_transform(grayscale_container_2darray))

    # Select HH sub-band for both

    stego_hh = numpy.copy(transformed_grayscale_stego)
    container_hh = numpy.copy(transformed_grayscale_container)

    # for row in range(0, int(size - 1)):
    #     for cell in range(0, int(size - 1)):
    #         stego_hh[row][cell] = transformed_grayscale_stego[row][cell]
    #
    # for row in range(0, int(size - 1)):
    #     for cell in range(0, int(size - 1)):
    #         container_hh[row][cell] = transformed_grayscale_container[row][cell]

    # Perform SVD

    hhu, stego_hhs, hhv = numpy.linalg.svd(stego_hh, full_matrices=True)
    hhu, container_hhs, hhv = numpy.linalg.svd(container_hh, full_matrices=True)

    # Define watermark singular values

    middle_sv_offset = 32

    watermark_s = numpy.zeros(watermark_size)

    for i in range(0, watermark_size):
        watermark_s[i] = (float(stego_hhs[i + middle_sv_offset]) - float(container_hhs[i + middle_sv_offset])) / (
            float(alpha) / float(numpy.amax(container_hhs)))

    # Construct the extracted watermark

    # print watermark_s

    wu, t, wv = svd(grayscale_watermark_2darray)

    extracted_watermark = numpy.dot(wu, numpy.dot(numpy.diag(watermark_s), wv))

    return extracted_watermark


def coarsest_level_extract(grayscale_stego_2darray, grayscale_container_2darray, grayscale_watermark_2darray,
                           watermark_size, alpha):
    # Get a 1-level 2D MR-WHT of Stego and Container

    transformed_grayscale_stego = multiresolutional_walsh_hadamard_transform(grayscale_stego_2darray)
    transformed_grayscale_container = multiresolutional_walsh_hadamard_transform(grayscale_container_2darray)

    # Select HH sub-band for both

    stego_hh = numpy.copy(transformed_grayscale_stego)
    container_hh = numpy.copy(transformed_grayscale_container)

    # for row in range(0, int(size - 1)):
    #     for cell in range(0, int(size - 1)):
    #         stego_hh[row][cell] = transformed_grayscale_stego[row][cell]
    #
    # for row in range(0, int(size - 1)):
    #     for cell in range(0, int(size - 1)):
    #         container_hh[row][cell] = transformed_grayscale_container[row][cell]

    # Perform SVD

    hhu, stego_hhs, hhv = numpy.linalg.svd(stego_hh, full_matrices=True)
    hhu, container_hhs, hhv = numpy.linalg.svd(container_hh, full_matrices=True)

    # Define watermark singular values

    middle_sv_offset = 63

    watermark_s = numpy.zeros(watermark_size)

    for i in range(0, watermark_size):
        watermark_s[i] = (float(stego_hhs[i + middle_sv_offset]) - float(container_hhs[i + middle_sv_offset])) / (
            float(alpha) / float(numpy.amax(container_hhs)))

    # Construct the extracted watermark

    # print watermark_s

    wu, t, wv = svd(grayscale_watermark_2darray)

    extracted_watermark = numpy.dot(wu, numpy.dot(numpy.diag(watermark_s), wv))

    return extracted_watermark


def multiresolutional_walsh_hadamard_transform(grayscale_image):
    """
    MR-WHT implementation.
    
    Outputs 1-level MR-WHT transformed image
    
    23-July-2015
    """

    def is_power2(num):
        return num != 0 and ((num & (num - 1)) == 0)

    assert is_power2(grayscale_image.shape[0]) and (
        grayscale_image.shape[0] == grayscale_image.shape[1]), 'Grayscale Image size is not a power of 2 or not square!'

    # image = Image.fromarray(numpy.uint8(grayscale_image))
    # image.show()

    # print 'Pre-transformed_grayscale_image:\r\n',grayscale_image

    # Do usual WHT
    n = grayscale_image.shape[0]
    h = hadamard(n)
    transformed_grayscale_image = numpy.dot(h, grayscale_image)

    # print 'transformed_grayscale_image:\r\n',transformed_grayscale_image

    # RowWiseTransformedGrayscaleImage = numpy.zeros(transformed_grayscale_image.shape)
    #
    # for row in range(0,n):
    #    for column in range(1,int(n/2)):
    #        RowWiseTransformedGrayscaleImage[row][column] = math.floor(float(transformed_grayscale_image[row][2*column-1]+transformed_grayscale_image[row][2*column])/2)
    #        RowWiseTransformedGrayscaleImage[row][column+int(n/2)-1] = transformed_grayscale_image[row][2*column-1]-transformed_grayscale_image[row][2*column]
    #
    # print 'Row wise transformed_grayscale_image:\r\n',RowWiseTransformedGrayscaleImage

    # ColumnWiseTransformedGrayscaleImage = numpy.zeros(transformed_grayscale_image.shape)
    #
    # for column in range(0,n):
    #    for row in range(1,int(n/2)):
    #        ColumnWiseTransformedGrayscaleImage[row][column] = math.floor(float(RowWiseTransformedGrayscaleImage[2*row-1][column]+RowWiseTransformedGrayscaleImage[2*row][column])/2)
    #        ColumnWiseTransformedGrayscaleImage[row+int(n/2)-1][column] = RowWiseTransformedGrayscaleImage[2*row-1][column]-RowWiseTransformedGrayscaleImage[2*row][column]
    #
    # print 'Column wise transformed_grayscale_image:\r\n',ColumnWiseTransformedGrayscaleImage

    # transformed_grayscale_image = RowWiseTransformedGrayscaleImage

    return transformed_grayscale_image


def inverse_multiresolutional_walsh_hadamard_transform(transformed_grayscale_image):
    """
    MR-WHT implementation.
    
    Outputs 1-level inverse MR-WHT transformed image
    
    23-July-2015
    """

    n = transformed_grayscale_image.shape[0]

    # ColumnWiseTransformedGrayscaleImage = numpy.zeros(transformed_grayscale_image.shape)
    #
    # for column in range(0,n):
    #    for row in range(1,int(n/2)):
    #        ColumnWiseTransformedGrayscaleImage[2*row-1][column] = transformed_grayscale_image[row][column] + math.floor(float(transformed_grayscale_image[n/2 + row][column] + 1)/2)
    #        ColumnWiseTransformedGrayscaleImage[2*row][column] = transformed_grayscale_image[2*row-1][column] - transformed_grayscale_image[n/2 + row][column]
    #        
    # print 'Column wise inverse_transformed_grayscale_image:\r\n',ColumnWiseTransformedGrayscaleImage

    # RowWiseTransformedGrayscaleImage = numpy.zeros(transformed_grayscale_image.shape)
    #
    # for row in range(0,n):
    #    for column in range(1,int(n/2)):
    #        RowWiseTransformedGrayscaleImage[row][2*column-1] = transformed_grayscale_image[row][column] + math.floor(float(transformed_grayscale_image[row][n/2 + column] + 1)/2)
    #        RowWiseTransformedGrayscaleImage[row][2*column] = transformed_grayscale_image[row][2*column-1] - transformed_grayscale_image[row][n/2 + column]
    #
    # print 'Row wise inverse_transformed_grayscale_image:\r\n',RowWiseTransformedGrayscaleImage

    # Do usual IWHT
    h = hadamard(n)
    inverse_transformed_grayscale_image = numpy.dot(h, transformed_grayscale_image) / n

    # image = Image.fromarray(numpy.uint8(inverse_transformed_grayscale_image))
    # image.show()

    return inverse_transformed_grayscale_image


def main(argv):
    mode = ''
    grayscale_container_path = ''
    grayscale_watermark_path = ''
    watermarked_image_path = ''
    extracted_coarsest_watermark_path = ''
    extracted_finest_watermark_path = ''
    watermark_size = 0
    alpha = 0

    try:
        opts, args = getopt.getopt(argv, "",
                                   ["mode=", "grayscale_container=", "grayscale_watermark=", "watermarked_image=",
                                    "extracted_coarsest_watermark=", "extracted_finest_watermark=", "watermark_size=",
                                    "alpha="])
    except getopt.GetoptError:
        print '\r\nPlease, use this software this way:'
        print 'Embedding: bhatnagar_embedding_method.py --mode embed --grayscale_container %path% ' \
              '--grayscale_watermark %path% --watermarked_image %path% --alpha %alpha%'
        print 'Extracting: bhatnagar_embedding_method.py --mode extract --watermarked_image %path% ' \
              '--grayscale_container %path% --grayscale_watermark %path% --extracted_coarsest_watermark %path% ' \
              '--extracted_finest_watermark %path% --watermark_size %size% --alpha %alpha%\r\n'
        sys.exit(2)

    for opt, arg in opts:
        if opt == '--mode':
            mode = arg
        elif opt == '--grayscale_container':
            grayscale_container_path = arg
        elif opt == '--grayscale_watermark':
            grayscale_watermark_path = arg
        elif opt == '--watermarked_image':
            watermarked_image_path = arg
        elif opt == '--extracted_coarsest_watermark':
            extracted_coarsest_watermark_path = arg
        elif opt == '--extracted_finest_watermark':
            extracted_finest_watermark_path = arg
        elif opt == '--watermark_size':
            watermark_size = int(arg)
        elif opt == '--alpha':
            alpha = float(arg)

    if mode == 'embed':
        if grayscale_container_path != "" and grayscale_watermark_path != "" and watermarked_image_path != "" \
                and alpha != 0:
            print '\r\nEmbedding started\r\n'
            bhatnagar_embed(grayscale_container_path, grayscale_watermark_path, watermarked_image_path, alpha)
            sys.exit(0)
    elif mode == 'extract':
        if watermarked_image_path != "" and grayscale_container_path != "" and grayscale_watermark_path != "" and extracted_coarsest_watermark_path != "" \
                and extracted_finest_watermark_path != "" and watermark_size != 0 and alpha != 0:
            print '\r\nExtracting started\r\n'
            bhatnagar_extract(watermarked_image_path, grayscale_container_path, grayscale_watermark_path,
                              extracted_coarsest_watermark_path,
                              extracted_finest_watermark_path, watermark_size, alpha)
            sys.exit(0)

    print '\r\nPlease, use this software this way:'
    print 'Embedding: bhatnagar_embedding_method.py --mode embed --grayscale_container %path% ' \
          '--grayscale_watermark %path% --watermarked_image %path% --alpha %alpha%'
    print 'Extracting: bhatnagar_embedding_method.py --mode extract --watermarked_image %path% ' \
          '--grayscale_container %path% --grayscale_watermark %path% --extracted_coarsest_watermark %path% ' \
          '--extracted_finest_watermark %path% --watermark_size %size% --alpha %alpha%\r\n'
    sys.exit(2)


if __name__ == "__main__":
    main(sys.argv[1:])
