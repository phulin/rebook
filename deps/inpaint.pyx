# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from __future__ import division
cimport numpy as np
import numpy as np

cimport cython

DTYPE = np.uint8

ctypedef np.uint8_t DTYPE_t

# IM = inpainting mask with 1s in background, 0s in foreground
def inpaint_ng14(np.ndarray[DTYPE_t, ndim=2] im,
                 np.ndarray[DTYPE_t, ndim=2] IM):
    cdef int im_h = im.shape[0]
    cdef int im_w = im.shape[1]

    cdef int y, x
    cdef int temp
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] I = \
        np.pad(im, (1, 1), 'edge')
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] Pi = \
        np.zeros([im_h, im_w], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] IM_padded = \
        np.pad(IM, (1, 1), 'constant', constant_values=1)
    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] M = IM_padded.copy()
    for y in range(1, im_h + 1):
        for x in range(1, im_w + 1):
            if M[y, x] == 0:
                temp  = I[y, x - 1] & -M[y, x - 1]
                temp += I[y - 1, x] & -M[y - 1, x]
                temp += I[y, x + 1] & -M[y, x + 1]
                temp += I[y + 1, x] & -M[y + 1, x]
                Pi[y - 1, x - 1] = temp / (M[y, x - 1] + M[y - 1, x] + \
                                           M[y, x + 1] + M[y + 1, x])
                I[y, x] = Pi[y - 1, x - 1]
                M[y, x] = 1

    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] Pmin = Pi.copy()
    cdef np.ndarray[np.uint32_t, ndim=2, mode="c"] Psum = Pi.copy().astype(np.uint32)

    np.copyto(M, IM_padded)
    for y in range(im_h, 0, -1):
        for x in range(1, im_w + 1):
            if M[y, x] == 0:
                temp  = I[y, x - 1] & -M[y, x - 1]
                temp += I[y - 1, x] & -M[y - 1, x]
                temp += I[y, x + 1] & -M[y, x + 1]
                temp += I[y + 1, x] & -M[y + 1, x]
                Pi[y - 1, x - 1] = temp / (M[y, x - 1] + M[y - 1, x] + \
                                           M[y, x + 1] + M[y + 1, x])
                I[y, x] = Pi[y - 1, x - 1]
                M[y, x] = 1
    np.minimum(Pmin, Pi, out=Pmin)
    Psum += Pi

    np.copyto(M, IM_padded)
    for y in range(1, im_h + 1):
        for x in range(im_w, 0, -1):
            if M[y, x] == 0:
                temp  = I[y, x - 1] & -M[y, x - 1]
                temp += I[y - 1, x] & -M[y - 1, x]
                temp += I[y, x + 1] & -M[y, x + 1]
                temp += I[y + 1, x] & -M[y + 1, x]
                Pi[y - 1, x - 1] = temp / (M[y, x - 1] + M[y - 1, x] + \
                                           M[y, x + 1] + M[y + 1, x])
                I[y, x] = Pi[y - 1, x - 1]
                M[y, x] = 1
    np.minimum(Pmin, Pi, out=Pmin)
    Psum += Pi

    np.copyto(M, IM_padded)
    for y in range(im_h, 0, -1):
        for x in range(im_w, 0, -1):
            if M[y, x] == 0:
                temp  = I[y, x - 1] & -M[y, x - 1]
                temp += I[y - 1, x] & -M[y - 1, x]
                temp += I[y, x + 1] & -M[y, x + 1]
                temp += I[y + 1, x] & -M[y + 1, x]
                Pi[y - 1, x - 1] = temp / (M[y, x - 1] + M[y - 1, x] + \
                                           M[y, x + 1] + M[y + 1, x])
                I[y, x] = Pi[y - 1, x - 1]
                M[y, x] = 1
    np.minimum(Pmin, Pi, out=Pmin)
    Psum += Pi

    cdef np.ndarray[DTYPE_t, ndim=2, mode="c"] Pavg = (Psum / 4).astype(np.uint8)

    return Pmin, Pavg, I[1:im_h + 1, 1:im_w + 1]
