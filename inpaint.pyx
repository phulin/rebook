import numpy as np
cimport numpy as np

cimport cython

DTYPE = np.uint8

ctypedef np.uint8_t DTYPE_t

# IM = inpainting mask with 1s in background, 0s in foreground
# @cython.boundscheck(False)
# @cython.wraparound(False)
def inpaint_lrtb(np.ndarray[DTYPE_t, ndim=2] im,
                 np.ndarray[DTYPE_t, ndim=2] IM):
    cdef int im_h = im.shape[0]
    cdef int im_w = im.shape[1]

    cdef int y, x
    cdef int temp
    cdef np.ndarray[DTYPE_t, ndim=2] I = im.copy()
    cdef np.ndarray[DTYPE_t, ndim=2] Pi = \
        np.zeros([im_h, im_w], dtype=DTYPE)

    cdef np.ndarray[DTYPE_t, ndim=2] M = IM.copy()
    for y in range(im_h):
        for x in range(im_w):
            if M[y][x] == 0:
                temp  = I[y][x - 1] & -M[y][x - 1]
                temp += I[y - 1][x] & -M[y - 1][x]
                temp += I[y][x + 1] & -M[y][x + 1]
                temp += I[y + 1][x] & -M[y + 1][x]
                Pi[y][x] = temp / (M[y][x - 1] + M[y - 1][x] + \
                                   M[y][x + 1] + M[y + 1][x])
                I[y][x] = Pi[y][x]
                M[y][x] = 1

    cdef np.ndarray[DTYPE_t, ndim=2] P = Pi.copy()

    M = IM.copy()
    for y in range(im_h - 1, -1, -1):
        for x in range(im_w):
            if M[y][x] == 0:
                temp  = I[y][x - 1] & -M[y][x - 1]
                temp += I[y - 1][x] & -M[y - 1][x]
                temp += I[y][x + 1] & -M[y][x + 1]
                temp += I[y + 1][x] & -M[y + 1][x]
                Pi[y][x] = temp / (M[y][x - 1] + M[y - 1][x] + \
                                   M[y][x + 1] + M[y + 1][x])
                I[y][x] = Pi[y][x]
                M[y][x] = 1
    np.minimum(P, Pi, out=P)

    M = IM.copy()
    for y in range(im_h):
        for x in range(im_w - 1, -1, -1):
            if M[y][x] == 0:
                temp  = I[y][x - 1] & -M[y][x - 1]
                temp += I[y - 1][x] & -M[y - 1][x]
                temp += I[y][x + 1] & -M[y][x + 1]
                temp += I[y + 1][x] & -M[y + 1][x]
                Pi[y][x] = temp / (M[y][x - 1] + M[y - 1][x] + \
                                   M[y][x + 1] + M[y + 1][x])
                I[y][x] = Pi[y][x]
                M[y][x] = 1
    np.minimum(P, Pi, out=P)

    M = IM.copy()
    for y in range(im_h - 1, -1, -1):
        for x in range(im_w -1, -1, -1):
            if M[y][x] == 0:
                temp  = I[y][x - 1] & -M[y][x - 1]
                temp += I[y - 1][x] & -M[y - 1][x]
                temp += I[y][x + 1] & -M[y][x + 1]
                temp += I[y + 1][x] & -M[y + 1][x]
                Pi[y][x] = temp / (M[y][x - 1] + M[y - 1][x] + \
                                   M[y][x + 1] + M[y + 1][x])
                I[y][x] = Pi[y][x]
                M[y][x] = 1
    np.minimum(P, Pi, out=P)

    return P
