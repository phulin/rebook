# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from __future__ import division
import numpy as np
cimport numpy as np

cimport cython

from numpy.polynomial.polynomial import Polynomial as Poly

def t_i_k(np.ndarray[np.float64_t, ndim=2] R,
          g,
          np.ndarray[np.float64_t, ndim=2] points,
          double t0):

    cdef np.ndarray[np.float64_t, ndim=1] ts, g_coef, gp_coef, ray, Of, ROf
    cdef np.ndarray[np.float64_t, ndim=2] rays

    rays = R.dot(points)
    cdef double f = 3270.5
    Of = np.array([0, 0, f], dtype=np.float64)
    ROf = R.dot(Of)
    cdef double ROf_x, ROf_y, ROf_z, Rp_x, Rp_y, Rp_z
    ROf_x, ROf_y, ROf_z = ROf
    ts = np.full(points.shape[1], t0)

    cdef int n = ts.shape[0]
    cdef int i, j, k, m
    cdef double t, y, yp, u, target, gp

    m = g.degree()

    # defer interior scaling until computation
    g_coef = g.coef.copy()
    g_coef[0] = 0
    gp_coef = g_coef[1:].copy()
    for k in range(1, m - 1):
        gp_coef[k] *= k + 1
    # print 'g :', g_coef
    # print 'g\':', gp_coef

    for i in range(n):
        Rp_x, Rp_y, Rp_z = rays[:, i]

        # solve: g([R(pt - Of)]_x) = [R(pt - Of)]_z
        #  g(ray[0] * t - ROf_x) = ray[2] * t - ROf_z
        #   h(t)  = g(Rp_x * t - ROf_x) - (Rp_z * t - ROf_z)
        #   h'(t) = g'(Rp_x * t - ROf_x) * Rp_x - Rp_z

        t = ts[i]
        for j in range(100):
            y = 0
            u = Rp_x * t - ROf_x
            for k in range(m, -1, -1):
                y *= u
                y += g_coef[k]
            target = Rp_z * t - ROf_z
            y -= target
            if abs(y) < 1e-5: break

            gp = 0
            for k in range(m - 1, -1, -1):
                gp *= u
                gp += gp_coef[k]
            yp = Rp_x * gp - Rp_z

            t -= y / yp

        # if j >= 99:
        #     print "warning iterations exceeded", y + target, target

        ts[i] = t

    # print 'final ts:', ts
    return ts, ((ts * rays).T - ROf).T
