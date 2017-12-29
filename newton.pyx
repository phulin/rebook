# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from __future__ import division
import numpy as np
cimport numpy as np

cimport cython

from numpy.polynomial.polynomial import Polynomial as Poly

def t_i_k(np.ndarray[np.float64_t, ndim=1] theta,
          np.ndarray[np.float64_t, ndim=2] R,
          g,
          np.ndarray[np.float64_t, ndim=2] points):

    cdef np.ndarray[np.float64_t, ndim=1] ts, g_ray_coef, g_rayp_coef, ray, Of, ROf
    cdef np.ndarray[np.float64_t, ndim=2] rays

    rays = R.dot(points)
    cdef double f = 3270.5
    Of = np.array([0, 0, f], dtype=np.float64)
    ROfx, ROfy, ROfz = R.dot(Of)
    ts = f / rays[2]

    cdef int n = ts.shape[0]
    cdef int i, j, k, m
    cdef double t0, t, y, yp, u

    m = g.degree()
    assert m <= 9

    for i in range(n):
        Rpx, Rpy, Rpz = rays[:, i]

        # solve: g([R(pt - Of)]_x) = [R(pt - Of)]_z
        #  g(ray[0] * t - ROfx) = ray[2] * t - ROfz
        #   h(t) = g(ray[0] * t - ROfx) - (ray[2] * t - ROfz)
        #   h'(t) = g'(ray[0] * t - ROfx) * ray[0] - ray[2]

        # defer interior scaling until computation
        g_ray_coef = g.coef.copy()
        g_rayp_coef = g_ray_coef[1:].copy()
        for k in range(1, m - 1):
            g_rayp_coef[k] *= m + 1
        # print g_ray_coef, g_rayp_coef

        t = ts[i]
        for j in range(100):
            y = 0
            u = Rpx * t - ROfx
            for k in range(m, -1, -1):
                y *= u
                y += g_ray_coef[k]
            y -= Rpz * t - ROfz
            if abs(y) < 1e-7: break

            # (g(tRpx - ROfx) - (tRpz - ROfz))' = g'(tRpx - ROfx) * Rpx - Rpz
            yp = 0
            for k in range(m - 1, -1, -1):
                yp *= u
                yp += g_rayp_coef[k]
            yp *= Rpx
            yp -= Rpz

            t -= y / yp

        if j >= 99: print "warning iterations exceeded", abs(y)
        # print j
        # print t
        ts[i] = t

    # print 'final ts:', ts
    return np.array(ts), rays
