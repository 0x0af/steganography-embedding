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
import numpy
from pylab import *

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
    return numpy.cumsum(x - x.mean(axis).reshape(*shp), axis)


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
    import numpy
    from numpy.polynomial import polynomial
    step = scales[0]
    out = nan + zeros((len(scales), x.shape[0] // step), 'f8')
    j = 0
    for scale in scales:
        if verbose: print '.', scale, step
        i0 = scale // 2 / step + 1
        y = rw(x[step - (scale // 2) % step:], scale, step)
        i = arange(scale)
        c = numpy.polynomial.polynomial.polyfit(i, y.T, m)
        rms = sqrt(((y - numpy.polynomial.polynomial.polyval(i, c)) ** 2).mean(1))
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
