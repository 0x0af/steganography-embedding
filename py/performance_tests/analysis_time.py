import time

import numpy
import scipy
from PIL import Image
from pylab import *
from scipy import io

from py.multifractal_analysis_methods.mfdfa import *


def calculate_cutoff_coefficients(multifractal, qstep=0.1, u_lim=15, l_lim=-15, cutoff_level=0.005):
    scales = numpy.floor(2.0 ** arange(1, 8.1, qstep)).astype('i4')
    RW = rwalk(multifractal.ravel())
    RMS = fast_rms(RW, scales, 1)

    qs = arange(l_lim + qstep, u_lim + qstep, qstep)
    Fq = comp_fq(RMS, qs)

    def mdfa(x, a_scales, a_qs):
        RW = rwalk(x)
        RMS = fast_rms(RW, a_scales)
        Fq = comp_fq(RMS, a_qs)
        Hq = numpy.zeros(len(a_qs), 'f8')
        for qi, q in enumerate(a_qs):
            C = polyfit(log2(a_scales), log2(Fq[:, qi]), 1)
            Hq[qi] = C[0]
            if abs(q - int(q)) > 0.1:
                continue
        tq = Hq * a_qs - 1
        hq = diff(tq) / (a_qs[1] - a_qs[0])
        Dq = (a_qs[:-1] * hq) - tq[:-1]
        return Fq, Hq, hq, tq, Dq

    Fq, Hq, hq, tq, Dq = mdfa(multifractal.ravel(), scales, qs)

    Hq_max = Hq.max()

    cutoff_hq_max = Hq[int(qs.size * 0.5) - 1]
    cutoff_q_max = qs[int(qs.size * 0.5) - 1]
    for index in range(int(qs.size * 0.5), qs.size):
        if abs(Hq[index] / Hq_max - cutoff_hq_max / Hq_max) / (qstep / (u_lim - l_lim)) < cutoff_level:
            cutoff_hq_max = Hq[index]
            cutoff_q_max = qs[index]
            break
        else:
            cutoff_hq_max = Hq[index]
            cutoff_q_max = qs[index]

    cutoff_hq_min = Hq[0]
    cutoff_q_min = qs[0]
    for index in range(0, int(qs.size * 0.5)):
        if abs(Hq[index] / Hq_max - cutoff_hq_min / Hq_max) / (qstep / (u_lim - l_lim)) > cutoff_level:
            cutoff_hq_min = Hq[index]
            cutoff_q_min = qs[index]
            break
        else:
            cutoff_hq_min = Hq[index]
            cutoff_q_min = qs[index]

    return cutoff_hq_max, cutoff_q_max, cutoff_hq_min, cutoff_q_min


def get_row_spectrum(multifractal, qstep=0.1, u_lim=15, l_lim=-15):
    scales = umath.floor(2.0 ** arange(1, 8.1, qstep)).astype('i4')
    RW = rwalk(multifractal.ravel())
    RMS = fast_rms(RW, scales, 1)

    qs = arange(l_lim + qstep, u_lim + qstep, qstep)
    Fq = comp_fq(RMS, qs)

    def mdfa(x, a_scales, a_qs):
        RW = rwalk(x)
        RMS = fast_rms(RW, a_scales)
        Fq = comp_fq(RMS, a_qs)
        Hq = numpy.zeros(len(a_qs), 'f8')
        for qi, q in enumerate(a_qs):
            C = polyfit(log2(a_scales), log2(Fq[:, qi]), 1)
            Hq[qi] = C[0]
            if abs(q - int(q)) > 0.1:
                continue
        tq = Hq * a_qs - 1
        hq = diff(tq) / (a_qs[1] - a_qs[0])
        Dq = (a_qs[:-1] * hq) - tq[:-1]
        return Fq, Hq, hq, tq, Dq

    Fq, Hq, hq, tq, Dq = mdfa(multifractal.ravel(), scales, qs)

    return Hq, qs


def analyze_image(image_name, image, skip):
    image.resize((512, 512), Image.ANTIALIAS)

    matrix_truecolor = numpy.array(image)

    spectrum = zeros((3, 512, 2, 300))  # 3 color channels, 512 rows, 2 lines of 300 values

    t0 = time.clock()

    for color_channel in range(0, matrix_truecolor.shape[2]):

        matrix_grayscale = matrix_truecolor[:, :, color_channel]

        for row_index in range(0, matrix_grayscale.shape[0], skip):
            Hq, qs = get_row_spectrum(matrix_grayscale[:, row_index])
            spectrum[color_channel][row_index][0] = Hq
            spectrum[color_channel][row_index][1] = qs

    timespan = time.clock() - t0

    scipy.io.savemat(image_name + '.mat',
                     mdict={'image_name': image_name, 'image': matrix_truecolor, 'spectrum': spectrum,
                            'timespan': timespan})

# def analyze_time:
# savefig(str(skip) + str(qstep) + '.png')
# matplotlib.pyplot.close()

# print skip, qstep, cutoff_hq_max, cutoff_q_max, cutoff_hq_min, cutoff_q_min, timespan

# show()

# for skippower in range(8, 0, -1):
#     skip = 2 ** skippower
#     for qstepfactor in range(2, 12, 2):
#         qstep = 1.0 / qstepfactor
#
#         t0 = time.clock()
#
#         mfr = zeros((lena.shape[0] * lena.shape[0] / skip, 1))
#
#         for row_index in range(lena.shape[0]):
#             if row_index % skip == 0 and mfr.shape[0] - (row_index / skip) * lena.shape[0] >= lena.shape[0]:
#                 mfr[(row_index / skip) * lena.shape[0]: (row_index / skip + 1) * lena.shape[0], 0] = lena[:,
#                                                                                                      row_index]
#         # figure()
#         # plot(mfr)
#         # show()
#
#         cutoff_hq_max, cutoff_q_max, cutoff_hq_min, cutoff_q_min = calculate_cutoff_coefficients(mfr, qstep, 15,
#                                                                                                  -15, 0.05)
#
#         timespan = time.clock() - t0
#
#         savefig(str(skip) + str(qstep) + '.png')
#         matplotlib.pyplot.close()
#
#         print skip, qstep, cutoff_hq_max, cutoff_q_max, cutoff_hq_min, cutoff_q_min, timespan
#
#         # show()
