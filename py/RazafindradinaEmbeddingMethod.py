# coding: utf-8


"""
Created on Thu Jul 23 2015

@author: Anton at 0x0af@ukr.net
"""

import numpy
import scipy
from scipy.fftpack import dct, idct
from PIL import Image


def schur_decomposition(grayscale_image_2darray):
    """
     Schur decomposition. Input a grayscale image to get it's upper triangular matrix and unitary matrix.
     
     Schur's theorem announced that : if A ∈ Cn×n, there exists a unitary matrix U and an upper triangular matrix T,
     such that: A = U × T × U'
     
     0x0af@ukr.net, 23-July-2015
    """

    triangular_matrix, unitary_matrix = scipy.linalg.schur(grayscale_image_2darray)

    return triangular_matrix, unitary_matrix


def inverse_schur_decomposition(triangular_matrix, unitary_matrix):
    """
     Inverse Schur decomposition. Input a 2D unitary matrix and a 2D triangular matrix to get the original grayscale image
    
     23-July-2015
    """

    original_grayscale_image_2darray = numpy.dot(unitary_matrix, numpy.dot(triangular_matrix, unitary_matrix.conj().T))

    return original_grayscale_image_2darray


def razafindradina_embed(grayscale_container_path, grayscale_watermark_path, watermarked_image_path, alpha):
    """    
    Razafindradina embedding method implementation. 
    
    Outputs the resulting watermarked image
    
    23-July-2015
    """

    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    grayscale_watermark_2darray = numpy.asarray(Image.open(grayscale_watermark_path).convert("L"))

    assert (grayscale_container_2darray.shape[0] == grayscale_container_2darray.shape[1]) and (
        grayscale_container_2darray.shape[0] == grayscale_watermark_2darray.shape[0]) and (
               grayscale_container_2darray.shape[1] == grayscale_watermark_2darray.shape[
                   1]), 'GrayscaleContainer and GrayscaleWatermark sizes do not match or not square'

    # Perform DCT on GrayscaleContainer

    # print grayscale_container_2darray

    gcdct = dct(dct(grayscale_container_2darray, axis=0, norm='ortho'), axis=1, norm='ortho')

    # print grayscale_container_2darray

    # Perform SchurDecomposition on GrayscaleWatermark

    gwsdt, gwsdu = schur_decomposition(grayscale_watermark_2darray)

    # alpha-blend GrayscaleWatermark TriangularMatrix into GrayscaleContainer DCT coeffs with alpha

    gcdct += gwsdt * alpha

    # Perform IDCT on GrayscaleContainer DCT coeffs to get WatermarkedImage

    watermarked_image_2darray = idct(idct(gcdct, axis=0, norm='ortho'), axis=1, norm='ortho')

    watermarked_image_2darray[watermarked_image_2darray > 255] = 255
    watermarked_image_2darray[watermarked_image_2darray < 0] = 0

    watermarked_image = Image.fromarray(numpy.uint8(watermarked_image_2darray))

    # watermarked_image.show()

    # Write image to file

    watermarked_image.save(watermarked_image_path)

    return


def razafindradina_extract(grayscale_stego_path, grayscale_container_path, grayscale_watermark_path,
                           extracted_grayscale_watermark_path, alpha):
    """
    Rezefindradina extracting method implementation.
    
    Outputs the extracted watermark
    
    23-July-2015
    """

    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    grayscale_stego_2darray = numpy.asarray(Image.open(grayscale_stego_path).convert("L"))
    grayscale_watermark_2darray = numpy.asarray(Image.open(grayscale_watermark_path).convert("L"))

    # Perform DCT on GrayscaleContainer

    gcdct = dct(dct(grayscale_container_2darray, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Perform DCT on GrayscaleStego

    gsdct = dct(dct(grayscale_stego_2darray, axis=0, norm='ortho'), axis=1, norm='ortho')

    # Extract ExtractedWatermarkTriangularMatrix with alpha

    ewsdt = (gsdct - gcdct) / alpha

    # Perform InverseSchurDecomposition on UnitaryMatrix (non-embedded) and TrinagularMatrix to get ExtractedGrayscaleWatermark

    gwsdt, gwsdu = schur_decomposition(grayscale_watermark_2darray)

    extracted_watermark_2darray = inverse_schur_decomposition(ewsdt, gwsdu)

    extracted_watermark_2darray[extracted_watermark_2darray > 255] = 255
    extracted_watermark_2darray[extracted_watermark_2darray < 0] = 0

    extracted_grayscale_watermark = Image.fromarray(numpy.uint8(extracted_watermark_2darray))
    # extracted_grayscale_watermark.show()

    # Write image to file

    extracted_grayscale_watermark.save(extracted_grayscale_watermark_path)

    return


razafindradina_embed("..\\steganography-embedding\\GrayscaleContainer.bmp",
                     "..\\steganography-embedding\\SameSizeGrayscaleWatermark.bmp",
                     "..\\steganography-embedding\\Razafindradina_Watermarked_Image.bmp",
                     0.01)

razafindradina_extract("..\\steganography-embedding\\Razafindradina_Watermarked_Image.bmp",
                       "..\\steganography-embedding\\GrayscaleContainer.bmp",
                       "..\\steganography-embedding\\SameSizeGrayscaleWatermark.bmp",
                       "..\\steganography-embedding\\Razafindradina_Extracted_Watermark.bmp",
                       0.01)
