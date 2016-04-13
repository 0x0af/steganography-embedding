import time

import numpy
import scipy
from pylab import *
from scipy import io


# def calculate_cutoff_coefficients(multifractal, qstep=0.1, u_lim=15, l_lim=-15, cutoff_level=0.005):
#     scales = numpy.floor(2.0 ** numpy.arange(1, 8.1, qstep)).astype('i4')
#     # RW = rwalk(multifractal.ravel())
#     # RMS = fast_rms(RW, scales, 1)
#
#     qs = arange(l_lim + qstep, u_lim + qstep, qstep)
#
#     # Fq = comp_fq(RMS, qs)
#
#     def mdfa(x, a_scales, a_qs):
#         RW = rwalk(x)
#         RMS = fast_rms(RW, a_scales)
#         Fq = comp_fq(RMS, a_qs)
#         Hq = numpy.zeros(len(a_qs), 'f8')
#         for qi, q in enumerate(a_qs):
#             C = polyfit(log2(a_scales), log2(Fq[:, qi]), 1)
#             Hq[qi] = C[0]
#             if abs(q - int(q)) > 0.1:
#                 continue
#         tq = Hq * a_qs - 1
#         hq = diff(tq) / (a_qs[1] - a_qs[0])
#         Dq = (a_qs[:-1] * hq) - tq[:-1]
#         return Fq, Hq, hq, tq, Dq
#
#     Fq, Hq, hq, tq, Dq = mdfa(multifractal.ravel(), scales, qs)
#
#     Hq_max = Hq.max()
#
#     cutoff_hq_max = Hq[int(qs.size * 0.5) - 1]
#     cutoff_q_max = qs[int(qs.size * 0.5) - 1]
#     for index in range(int(qs.size * 0.5), qs.size):
#         if abs(Hq[index] / Hq_max - cutoff_hq_max / Hq_max) / (qstep / (u_lim - l_lim)) < cutoff_level:
#             cutoff_hq_max = Hq[index]
#             cutoff_q_max = qs[index]
#             break
#         else:
#             cutoff_hq_max = Hq[index]
#             cutoff_q_max = qs[index]
#
#     cutoff_hq_min = Hq[0]
#     cutoff_q_min = qs[0]
#     for index in range(0, int(qs.size * 0.5)):
#         if abs(Hq[index] / Hq_max - cutoff_hq_min / Hq_max) / (qstep / (u_lim - l_lim)) > cutoff_level:
#             cutoff_hq_min = Hq[index]
#             cutoff_q_min = qs[index]
#             break
#         else:
#             cutoff_hq_min = Hq[index]
#             cutoff_q_min = qs[index]
#
#     return cutoff_hq_max, cutoff_q_max, cutoff_hq_min, cutoff_q_min


def get_row_spectrum(signal, qstep=0.1, u_lim=15, l_lim=-15):
    scaling_window_sizes = numpy.floor(2.0 ** numpy.arange(1, 9)).astype('i4')

    q = numpy.arange(l_lim + qstep, u_lim + qstep, qstep)

    cumulative_sums = numpy.cumsum(signal - signal.mean())

    local_linear_trends = zeros((scaling_window_sizes.shape[0], 2 * floor(signal.shape[0] / min(scaling_window_sizes))))

    print local_linear_trends.shape

    for i in range(0, scaling_window_sizes.shape[0]):
        for j in range(0, int(floor(signal.shape[0] / scaling_window_sizes[i]))):
            current_position_left = numpy.arange(j * scaling_window_sizes[i], (j + 1) * scaling_window_sizes[i] - 1)
            current_position_right = numpy.arange(signal.shape[0] - j * scaling_window_sizes[i] - 1,
                                                  signal.shape[0] - (j + 1) * scaling_window_sizes[i], -1)
            left_fit = polyfit(current_position_left, cumulative_sums[current_position_left], 1)
            right_fit = polyfit(current_position_right, cumulative_sums[current_position_right], 1)
            left_curve = polyval(left_fit, current_position_left)
            right_curve = polyval(right_fit, current_position_right)
            local_linear_trends[i, j] = sqrt(mean((cumulative_sums[current_position_left] - left_curve) ** 2))
            local_linear_trends[i, j + floor(signal.shape[0] / scaling_window_sizes[i])] = sqrt(
                mean((cumulative_sums[current_position_right] - right_curve) ** 2))

    Hq = zeros(q.shape)

    for k in range(0, q.shape[0]):
        fluctuation_function = zeros(scaling_window_sizes.shape)
        for i in range(0, scaling_window_sizes.shape[0]):
            trend = local_linear_trends[i, local_linear_trends[i, :] != 0]
            if q[k] != 0:
                fluctuation_function[i] = mean(trend ** q[k]) ** (1 / q[k])
            else:
                fluctuation_function[i] = exp(0.5 * mean(log(trend ** 2)))
        try:
            not_nan_positions = numpy.invert(isnan(log(fluctuation_function))) * \
                                numpy.invert(isinf(log(fluctuation_function)))
        finally:
            fit = polyfit(log(scaling_window_sizes[not_nan_positions]), log(fluctuation_function[not_nan_positions]), 1)
            Hq[k] = fit[0]

    tq = Hq * q - 1
    hq = diff(tq) / qstep
    Dq = (q[:-1] * hq) - tq[:-1]

    Dq.resize((1, 300))
    hq.resize((1, 300))

    return Hq, q, Dq, hq


def analyze_image(image_name, image, skip):
    matrix_truecolor = image

    spectrum = numpy.zeros((3, 512, 4, 300))  # 3 color channels, 512 rows, 4 lines of 300 values

    t0 = time.clock()

    for color_channel in range(0, matrix_truecolor.shape[2]):

        matrix_grayscale = matrix_truecolor[:, :, color_channel]

        for row_index in range(0, matrix_grayscale.shape[0], skip):
            Hq, qs, Dq, hq = get_row_spectrum(matrix_grayscale[:, row_index])
            spectrum[color_channel][row_index][0] = Hq
            spectrum[color_channel][row_index][1] = qs
            spectrum[color_channel][row_index][2] = Dq
            spectrum[color_channel][row_index][3] = hq

    timespan = time.clock() - t0

    scipy.io.savemat(image_name + '.mat',
                     mdict={'image_name': image_name, 'image': matrix_truecolor, 'spectrum': spectrum,
                            'timespan': timespan})
