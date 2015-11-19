# ----------------------------------------------------------------------------
# pymdfa
#
# Copyright (c) 2014 Peter Jurica @ RIKEN Brain Science Institute, Japan
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
# ----------------------------------------------------------------------------

"""
The 'pymdfa' module
-------------------
A minimalistic and fast implementation of MFDFA in Python.

Main functions:

 * compRMS - computes RMS for desired scales, returns 2D array of RMS for each scale, uncomputable elements contain "nan"
 * fastRMS - computes RMS for desired scales, fast vectorized version, returns 2D array of RMS for each scale, uncomputable elements contain "nan"
 * simpleRMS - computes RMS for desired scales, fast vectorized version, returns a list of RMS for each scale
 * compFq - computes F

Helpers:

* rw - transforms timeseries (vector array) into a matrix of running windows without copying data
* rwalk - subtracts mean and return cumulative sum

Citation:

    Jurica P. Multifractal analysis for all. Frontiers in Physiology. 2015;6:27. doi:10.3389/fphys.2015.00027.

"""
from numpy import ma, polyval, nan
from numpy.core.umath import isnan
from numpy.ma import cumsum, arange, zeros, polyfit, sqrt, exp, log

__all__ = ["fast_rms", "comp_rms", "comp_fq", "simple_rms", "rwalk"]


def rw(x, w, step=1):
    from numpy.lib.stride_tricks import as_strided as ast
    if not x.flags['C_CONTIGUOUS']:
        x = x.copy()
    if hasattr(x, 'mask'):
        return ma.array(ast(x.data, ((x.shape[0] - w) // step + 1, w), ((step * x.dtype.itemsize), x.dtype.itemsize)),
                        mask=ast(x.mask, ((x.shape[0] - w) // step + 1, w),
                                 ((step * x.mask.dtype.itemsize), x.mask.dtype.itemsize)))
    else:
        return ast(x, ((x.shape[0] - w) // step + 1, w), ((step * x.dtype.itemsize), x.dtype.itemsize))


def rwalk(x, axis=-1):
    shp = list(x.shape)
    shp[axis] = 1
    return cumsum(x - x.mean(axis).reshape(*shp), axis)


def comp_rms(x, scales, m=1, verbose=False):
    t = arange(x.shape[0])
    step = scales[0]
    i0s = arange(0, x.shape[0], step)
    out = zeros((len(scales), i0s.shape[0]), 'f8')
    for si, scale in enumerate(scales):
        if verbose:
            print '.',
        s2 = scale // 2
        for j, i0 in enumerate(i0s - s2):
            i1 = i0 + scale
            if i0 < 0 or i1 >= x.shape[0]:
                out[si, j] = nan
                continue
            t0 = t[i0:i1]
            C = polyfit(t0, x[i0:i1], m)
            fit = polyval(C, t0)
            out[si, j] = sqrt(((x[i0:i1] - fit) ** 2).mean())
    return out


def simple_rms(x, scales, m=1, verbose=False):
    out = []
    for scale in scales:
        y = rw(x, scale, scale)
        i = arange(scale)
        c = polyfit(i, y.T, m)
        out.append(sqrt(((y - polyval(i, c)) ** 2).mean(1)))
    return out


def fast_rms(x, scales, m=1, verbose=False):
    step = scales[0]
    out = nan + zeros((len(scales), x.shape[0] // step), 'f8')
    j = 0
    for scale in scales:
        if verbose: print '.', scale, step
        i0 = scale // 2 / step + 1
        y = rw(x[step - (scale // 2) % step:], scale, step)
        i = arange(scale)
        c = polyfit(i, y.T, m)
        rms = sqrt(((y - polyval(i, c)) ** 2).mean(1))
        out[j, i0:i0 + rms.shape[0]] = rms
        j += 1
    return out


def comp_fq(rms, qs):
    out = zeros((rms.shape[0], len(qs)), 'f8')
    m_rms = ma.array(rms, mask=isnan(rms))
    for qi in xrange(len(qs)):
        p = qs[qi]
        out[:, qi] = (m_rms ** p).mean(1) ** (1.0 / p)
    out[:, qs == 0] = exp(0.5 * (log(m_rms ** 2.0)).mean(1))[:, None]
    return out

# def demo():
#     import os
#     import time
#     rcParams['figure.figsize'] = (14, 8)
#     from scipy.io import loadmat
#
#     fname = 'fractaldata.mat'
#     if not os.path.exists(fname):
#         from urllib import urlretrieve
#         print 'Downloading %s.' % fname
#         urlretrieve('http://bsp.brain.riken.jp/~juricap/mdfa/%s' % fname, fname)
#
#     o = loadmat('fractaldata.mat')
#     whitenoise = o['whitenoise']
#     monofractal = o['monofractal']
#     multifractal = o['multifractal']
#
#     scstep = 8
#     scales = floor(2.0 ** arange(4, 10.1, 1.0 / scstep)).astype('i4')
#     RW = rwalk(multifractal.ravel())
#     t0 = time.clock()
#     RMS0 = compRMS(RW, scales, 1)
#     dtslow = time.clock() - t0
#     print 'compRMS took %0.3fs' % dtslow
#     t0 = time.clock()
#     RMS = fastRMS(RW, scales, 1)
#     dtfast = time.clock() - t0
#     print 'fast RMS took %0.3fs' % dtfast
#
#     figure()
#     subplot(211)
#     t = arange(0, RW.shape[0], scales[0]) + scales[0] / 2.0
#     imshow(RMS0, extent=(t[0], t[-1], log2(scales[0]), log2(scales[-1])), aspect='auto')
#     yticks(log2(scales)[::scstep], scales[::scstep])
#     text(500, log2(scales[-scstep]), 'compRMS (%0.3fs)' % dtslow, ha='left', color='w', fontsize=20)
#     ylabel('Scale')
#     colorbar()
#     subplot(212)
#     imshow(RMS, extent=(t[0], t[-1], log2(scales[0]), log2(scales[-1])), aspect='auto')
#     yticks(log2(scales)[::scstep], scales[::scstep])
#     text(500, log2(scales[-scstep]), 'fastRMS (%0.3fs)' % dtfast, ha='left', color='w', fontsize=20)
#     xlabel('Sample index')
#     ylabel('Scale')
#     colorbar()
#
#     # The output of **fastRMS** gives enough points for smoots MFDFA spectra.
#     qstep = 4
#     qs = arange(-5, 5.01, 1.0 / qstep)
#     Fq = compFq(RMS, qs)
#
#     def show_fits(scales, Fq):
#         plot(scales[::4], Fq[::4, ::4], '.-', lw=0.1)
#         gca().set_xscale('log')
#         gca().set_yscale('log')
#         margins(0, 0)
#         xticks(scales[::8], scales[::8])
#         yticks(2.0 ** arange(-4, 6), 2.0 ** arange(-4, 6))
#         xlabel('scale')
#         ylabel('Fq')
#
#     def MDFA(X, scales, qs):
#         RW = rwalk(X)
#         RMS = fastRMS(RW, scales)
#         Fq = compFq(RMS, qs)
#         Hq = zeros(len(qs), 'f8')
#         for qi, q in enumerate(qs):
#             C = polyfit(log2(scales), log2(Fq[:, qi]), 1)
#             Hq[qi] = C[0]
#             if abs(q - int(q)) > 0.1: continue
#             loglog(scales, 2 ** polyval(C, log2(scales)), lw=0.5, label='q=%d [H=%0.2f]' % (q, Hq[qi]))
#         tq = Hq * qs - 1
#         hq = diff(tq) / (qs[1] - qs[0])
#         Dq = (qs[:-1] * hq) - tq[:-1]
#         return Fq, Hq, hq, tq, Dq
#
#     figure()
#     subplot(231)
#     Fq, Hq, hq, tq, Dq = MDFA(multifractal.ravel(), scales, qs)
#     show_fits(scales, Fq)
#     yl = ylim()
#     subplot(223)
#     plot(qs, Hq, '-')
#     subplot(224)
#     plot(hq, Dq, '.-')
#
#     subplot(232)
#     Fq, Hq, hq, tq, Dq = MDFA(monofractal.ravel(), scales, qs)
#     show_fits(scales, Fq)
#     ylim(yl)
#     subplot(223)
#     plot(qs, Hq, '-')
#     subplot(224)
#     plot(hq, Dq, '.-')
#
#     subplot(233)
#     Fq, Hq, hq, tq, Dq = MDFA(whitenoise.ravel(), scales, qs)
#     show_fits(scales, Fq)
#     ylim(yl)
#     subplot(223)
#     plot(qs, Hq, '-')
#     subplot(224)
#     plot(hq, Dq, '.-')
#
#     subplot(223)
#     xlabel('q')
#     ylabel('Hq')
#     subplot(224)
#     xlabel('hq')
#     ylabel('Dq')
#
#     subplot(223)
#     legend(['Multifractal', 'Monofractal', 'White noise'])
#
#
# if __name__ == "__main__":
#     from numpy import *
#     from pylab import *
#
#     demo()
#     show()
