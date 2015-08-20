# coding: utf-8


"""
Created on Thu Jul 23 2015

@author: Anton at 0x0af@ukr.net
"""

import math

import numpy
import pywt
from PIL import Image


def dhanalakshmi_embed(grayscale_container_path, grayscale_primary_watermark_path, grayscale_secondary_watermark_path,
                       watermarked_image_path, alpha1, alpha2):
    """    
    Dhanalakshmi embedding method implementation. 
    
    Outputs the resulting watermarked image
    
    23-July-2015
    """

    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    grayscale_primary_watermark_2darray = numpy.asarray(Image.open(grayscale_primary_watermark_path).convert("L"))
    grayscale_secondary_watermark_2darray = numpy.asarray(Image.open(grayscale_secondary_watermark_path).convert("L"))

    # print grayscale_container_2darray.shape

    # Perform DWT on PrimaryWatermark

    gpwll, (gpwlh, gpwhl, gpwhh) = pywt.dwt2(grayscale_primary_watermark_2darray, 'haar')

    # Perform SVD on PrimaryWatermark DWT coeffs

    gpwllu, gpwlls, gpwllv = numpy.linalg.svd(gpwll, full_matrices=True)
    gpwlhu, gpwlhs, gpwlhv = numpy.linalg.svd(gpwlh, full_matrices=True)
    gpwhlu, gpwhls, gpwhlv = numpy.linalg.svd(gpwhl, full_matrices=True)
    gpwhhu, gpwhhs, gpwhhv = numpy.linalg.svd(gpwhh, full_matrices=True)

    # print gpwlls.shape,gpwlhs.shape,gpwhls.shape,gpwhhs.shape

    # Perform SVD on SecondaryWatermark

    gswu, gsws, gswv = numpy.linalg.svd(grayscale_secondary_watermark_2darray, full_matrices=True)

    # print gsws.shape

    # Alpha-blend SecondaryWatermark singular values into PrimaryWatermark DWT coeffs singular values

    gpwlls += alpha1 * gsws
    gpwlhs += alpha1 * gsws
    gpwhls += alpha1 * gsws
    gpwhhs += alpha1 * gsws

    # Perform LogisticMapChaoticEncryption on PrimaryWatermark DWT coeffs singular values

    epwlls, epwlhs, epwhls, epwhhs = logistic_map_chaotic_encryption(gpwlls, gpwlhs, gpwhls, gpwhhs)

    # Break GrayscaleContainer into non-overlapping blocks of PrimaryWatermark's shape an run for each

    watermarked_image_2darray = numpy.copy(grayscale_container_2darray)

    block = grayscale_primary_watermark_2darray.shape[0]
    cols = int(math.floor(grayscale_container_2darray.shape[0] / grayscale_primary_watermark_2darray.shape[0]))
    rows = int(math.floor(grayscale_container_2darray.shape[1] / grayscale_primary_watermark_2darray.shape[1]))

    for row in range(1, rows + 1):
        for col in range(1, cols + 1):
            # Perform DWT on GrayscaleContainer's block

            gcbll, (gcblh, gcbhl, gcbhh) = pywt.dwt2(
                grayscale_container_2darray[(row - 1) * block:row * block, (col - 1) * block:col * block], 'haar')

            # Perform SVD on GrayscaleContainer's block DWT coeffs

            gcbllu, gcblls, gcbllv = numpy.linalg.svd(gcbll, full_matrices=True)
            gcblhu, gcblhs, gcblhv = numpy.linalg.svd(gcblh, full_matrices=True)
            gcbhlu, gcbhls, gcbhlv = numpy.linalg.svd(gcbhl, full_matrices=True)
            gcbhhu, gcbhhs, gcbhhv = numpy.linalg.svd(gcbhh, full_matrices=True)

            # Alpha-blend EncryptedPrimaryWatermark singular values into GrayscaleContainer's block DWT coeffs singular values

            gcblls += alpha2 * epwlls
            gcblhs += alpha2 * epwlhs
            gcbhls += alpha2 * epwhls
            gcbhhs += alpha2 * epwhhs

            # Perform ISVD to get the WatermarkedImage's block DWT coeffs

            gcbll = numpy.dot(gcbllu, numpy.dot(numpy.diag(gcblls), gcbllv))
            gcblh = numpy.dot(gcblhu, numpy.dot(numpy.diag(gcblhs), gcblhv))
            gcbhl = numpy.dot(gcbhlu, numpy.dot(numpy.diag(gcbhls), gcbhlv))
            gcbhh = numpy.dot(gcbhhu, numpy.dot(numpy.diag(gcbhhs), gcbhhv))

            # Perform IDWT to get the WatermarkedImage's block

            watermarked_image_block = pywt.idwt2((gcbll, (gcblh, gcbhl, gcbhh)), 'haar')

            # Put WatermarkedImage's block into WatermarkedImage

            watermarked_image_2darray[(row - 1) * block:row * block,
            (col - 1) * block:col * block] = watermarked_image_block

    watermarked_image_2darray[watermarked_image_2darray > 255] = 255
    watermarked_image_2darray[watermarked_image_2darray < 0] = 0

    watermarked_image = Image.fromarray(numpy.uint8(watermarked_image_2darray))
    # watermarked_image.show()

    # Write image to file

    watermarked_image.save(watermarked_image_path)

    return


def dhanalakshmi_extract(grayscale_stego_path, grayscale_container_path, grayscale_primary_watermark_path,
                         grayscale_secondary_watermark_path, grayscale_primary_extracted_watermark_path,
                         grayscale_secondary_extracted_watermark_path, alpha1, alpha2):
    """
    Dhanalakshmi extracting method implementation.
    
    Outputs the extracted watermark
    
    23-July-2015
    """

    grayscale_stego_2darray = numpy.asarray(Image.open(grayscale_stego_path).convert("L"))
    grayscale_container_2darray = numpy.asarray(Image.open(grayscale_container_path).convert("L"))
    grayscale_primary_watermark2darray = numpy.asarray(Image.open(grayscale_primary_watermark_path).convert("L"))
    grayscale_secondary_watermark2darray = numpy.asarray(Image.open(grayscale_secondary_watermark_path).convert("L"))

    # TODO: average through blocks for more robustness

    block = grayscale_primary_watermark2darray.shape[0]
    cols = int(math.floor(grayscale_container_2darray.shape[0] / grayscale_primary_watermark2darray.shape[0]))
    rows = int(math.floor(grayscale_container_2darray.shape[1] / grayscale_primary_watermark2darray.shape[1]))

    row = int(rows / 2 + 1)
    col = int(cols / 2 + 1)

    # Perform DWT on GrayscaleStego's block

    gsbll, (gsblh, gsbhl, gsbhh) = pywt.dwt2(
        grayscale_stego_2darray[(row - 1) * block:row * block, (col - 1) * block:col * block], 'haar')

    # Perform SVD on GrayscaleStego's block DWT coeffs

    gsbllu, gsblls, gsbllv = numpy.linalg.svd(gsbll, full_matrices=True)
    gsblhu, gsblhs, gsblhv = numpy.linalg.svd(gsblh, full_matrices=True)
    gsbhlu, gsbhls, gsbhlv = numpy.linalg.svd(gsbhl, full_matrices=True)
    gsbhhu, gsbhhs, gsbhhv = numpy.linalg.svd(gsbhh, full_matrices=True)

    # Perform DWT on GrayscaleContainer's block

    gcbll, (gcblh, gcbhl, gcbhh) = pywt.dwt2(
        grayscale_container_2darray[(row - 1) * block:row * block, (col - 1) * block:col * block], 'haar')

    # Perform SVD on GrayscaleContainer's block DWT coeffs

    gcbllu, gcblls, gcbllv = numpy.linalg.svd(gcbll, full_matrices=True)
    gcblhu, gcblhs, gcblhv = numpy.linalg.svd(gcblh, full_matrices=True)
    gcbhlu, gcbhls, gcbhlv = numpy.linalg.svd(gcbhl, full_matrices=True)
    gcbhhu, gcbhhs, gcbhhv = numpy.linalg.svd(gcbhh, full_matrices=True)

    # Extract EncryptedPrimaryWatermark DWT coeffs singular values with alpha2

    epwlls = (gsblls - gcblls) / alpha2
    epwlhs = (gsblhs - gcblhs) / alpha2
    epwhls = (gsbhls - gcbhls) / alpha2
    epwhhs = (gsbhhs - gcbhhs) / alpha2

    # Perform LogisticMapChaoticDecryption on EncryptedPrimaryWatermark DWT coeffs singular values

    dpwlls, dpwlhs, dpwhls, dpwhhs = logistic_map_chaotic_decryption(epwlls, epwlhs, epwhls, epwhhs)

    # Perform DWT on PrimaryWatermark (non-embedded)

    gpwll, (gpwlh, gpwhl, gpwhh) = pywt.dwt2(grayscale_primary_watermark2darray, 'haar')

    # Perform SVD on PrimaryWatermark (non-embedded) DWT coeffs

    gpwllu, gpwlls, gpwllv = numpy.linalg.svd(gpwll, full_matrices=True)
    gpwlhu, gpwlhs, gpwlhv = numpy.linalg.svd(gpwlh, full_matrices=True)
    gpwhlu, gpwhls, gpwhlv = numpy.linalg.svd(gpwhl, full_matrices=True)
    gpwhhu, gpwhhs, gpwhhv = numpy.linalg.svd(gpwhh, full_matrices=True)

    # print gpwlls.shape,gpwlhs.shape,gpwhls.shape,gpwhhs.shape
    # print dpwlls.shape,dpwlhs.shape,dpwhls.shape,dpwhhs.shape

    # Perform ISVD on decrypted_primary_watermark DWT coeffs singular values

    dpwll = numpy.dot(gpwllu, numpy.dot(numpy.diag(dpwlls), gpwllv))
    dpwlh = numpy.dot(gpwlhu, numpy.dot(numpy.diag(dpwlhs), gpwlhv))
    dpwhl = numpy.dot(gpwhlu, numpy.dot(numpy.diag(dpwhls), gpwhlv))
    dpwhh = numpy.dot(gpwhhu, numpy.dot(numpy.diag(dpwhhs), gpwhhv))

    # Perform IDWT on decrypted_primary_watermark DWT coeffs

    decrypted_primary_watermark_2darray = pywt.idwt2((dpwll, (dpwlh, dpwhl, dpwhh)), 'haar')

    decrypted_primary_watermark_2darray[decrypted_primary_watermark_2darray > 255] = 255
    decrypted_primary_watermark_2darray[decrypted_primary_watermark_2darray < 0] = 0

    decrypted_primary_watermark = Image.fromarray(numpy.uint8(decrypted_primary_watermark_2darray))
    # decrypted_primary_watermark.show()

    # Write decrypted_primary_watermark to file

    decrypted_primary_watermark.save(grayscale_primary_extracted_watermark_path)

    # Extract secondary_watermark singular values with alpha1
    # TODO: extract from all levels and average for more robustness

    sws = (dpwlls - gpwlls) / alpha1

    # Perform SVD on secondary_watermark (non-embedded)

    gswu, gsws, gswv = numpy.linalg.svd(grayscale_secondary_watermark2darray, full_matrices=True)

    # Perform ISVD to get secondary_watermark

    sw2darray = numpy.dot(gswu, numpy.dot(numpy.diag(sws), gswv))

    sw2darray[sw2darray > 255] = 255
    sw2darray[sw2darray < 0] = 0

    secondary_watermark = Image.fromarray(numpy.uint8(sw2darray))
    # secondary_watermark.show()

    # Write decrypted_primary_watermark to file

    secondary_watermark.save(grayscale_secondary_extracted_watermark_path)

    return


def logistic_map_chaotic_encryption(ll, hl, lh, hh):
    """
    Logistic map chaotic encryption implementation.
    
    Outputs encrypted watermark
    
    23-July-2015
    """

    # TODO: implement logistic map chaotic encryption

    return ll, hl, lh, hh  # some key?


def logistic_map_chaotic_decryption(ll, hl, lh, hh):  # some key?
    """
    Logistic map chaotic decryption implementation.
    
    Outputs original watermark
    
    23-July-2015
    """

    # TODO: implement logistic map chaotic decryption

    return ll, hl, lh, hh


dhanalakshmi_embed("..\\steganography-embedding\\GrayscaleContainer.bmp",
                   "..\\steganography-embedding\\GrayscaleWatermark.bmp",
                   "..\\steganography-embedding\\GrayscaleSecondaryWatermark.bmp",
                   "..\\steganography-embedding\\Dhanalakshmi_Watermarked_Image.bmp", 0.01,
                   0.01)

dhanalakshmi_extract("..\\steganography-embedding\\Dhanalakshmi_Watermarked_Image.bmp",
                     "..\\steganography-embedding\\GrayscaleContainer.bmp",
                     "..\\steganography-embedding\\GrayscaleWatermark.bmp",
                     "..\\steganography-embedding\\GrayscaleSecondaryWatermark.bmp",
                     "..\\steganography-embedding\\Dhanalakshmi_Extracted_Primary_Watermark.bmp",
                     "..\\steganography-embedding\\Dhanalakshmi_Extracted_Secondary_Watermark.bmp",
                     0.01, 0.01)
