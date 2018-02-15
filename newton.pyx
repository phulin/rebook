# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from __future__ import division
import numpy as np
cimport numpy as np
cimport libc.math
from libc.math cimport fabs, fma, INFINITY, isfinite

cimport cython

from numpy.polynomial.polynomial import Polynomial as Poly

cdef double poly_eval_verbose(np.ndarray[np.float64_t, ndim=1] coef, double x):
    cdef double y = 0
    cdef int k, m = coef.shape[0] - 1
    for k in range(m, -1, -1):
        y = fma(y, x, coef[k])  # y * x + a_k
        print k, y, x, coef[k]

    return y

cdef double poly_eval(np.ndarray[np.float64_t, ndim=1] coef, double x):
    cdef double y = 0
    cdef int k, m = coef.shape[0] - 1
    for k in range(m, -1, -1):
        y = fma(y, x, coef[k])  # y * x + a_k

    return y

# g(x) = 1/w h(wx)
# g'(x) = h'(wx)
def t_i_k(np.ndarray[np.float64_t, ndim=2] R,
          g,
          np.ndarray[np.float64_t, ndim=2] points,
          np.ndarray[np.float64_t, ndim=1] t0s):

    cdef np.ndarray[np.float64_t, ndim=1] ts, h_coef, hp_coef, ray, Of, ROf
    cdef np.ndarray[np.float64_t, ndim=1] roots_t
    cdef np.ndarray roots_u
    cdef np.ndarray[np.float64_t, ndim=2] rays

    rays = R.dot(points)
    cdef double f = 3270.5
    Of = np.array([0, 0, f], dtype=np.float64)
    ROf = R.dot(Of)
    cdef double ROf_x, ROf_y, ROf_z, Rp_x, Rp_y, Rp_z
    ROf_x, ROf_y, ROf_z = ROf
    ts = np.zeros((points.shape[1],))
    assert ts.shape[0] == t0s.shape[0]

    cdef int n = ts.shape[0]
    cdef int i, j, k, m, big_ys, done
    cdef double t, y, yp, u, gp, best_t, t0
    cdef double u_minus, u_plus, y_minus, y_plus

    cdef double w = g.omega

    # defer interior scaling until computation
    m = g.degree()
    h_coef = g.h.coef.copy()
    h_coef[0] = 0
    hp_coef = h_coef[1:] * np.arange(1, g.degree() + 1)

    if np.all(h_coef == 0.):
        ts = ROf_z / rays[2]
        return ts, ts * rays - ROf[:, np.newaxis]

    big_ys = 0

    for i in range(n):
        Rp_x = rays[0, i]
        Rp_y = rays[1, i]
        Rp_z = rays[2, i]

        # solve: g([R(pt - Of)]_x) = [R(pt - Of)]_z
        #  g(ray[0] * t - ROf_x) = ray[2] * t - ROf_z
        #  g(t) = h(w * t) / w
        #  g'(t) = h'(w * t)
        #
        #   u = w * (Rp_x * t - ROf_x); t = (u / w + ROf_x) / Rp_x
        #
        #   s(t)  = g(Rp_x * t - ROf_x) - (Rp_z * t - ROf_z)
        #         = h(u) / w - (Rp_z * t - ROf_z)
        #   s'(t) = g'(Rp_x * t - ROf_x) * Rp_x - Rp_z
        #         = h'(u) * Rp_x - Rp_z
        #
        #   s(u)  = h(u) / w - (Rp_z / Rp_x * (u / w + ROf_x) - ROf_z)

        t0 = t0s[i]
        done = False
        if isfinite(t0):
            t = t0
            for j in range(20):
                u = w * (Rp_x * t - ROf_x)
                y = poly_eval(h_coef, u) / w - fma(Rp_z, t, -ROf_z)
                if fabs(y) < 1e-8:
                    # print j
                    break
                yp = fma(poly_eval(hp_coef, u), Rp_x, -Rp_z)
                t -= y / yp

            u_minus = w * (Rp_x * t * 0.99 - ROf_x)
            y_minus = poly_eval(h_coef, u_minus) / w - fma(Rp_z, t * 0.99, -ROf_z)
            u_plus = w * (Rp_x * t * 0.01 - ROf_x)
            y_plus = poly_eval(h_coef, u_plus) / w - fma(Rp_z, t * 0.01, -ROf_z)
            if y_minus * y_plus > 0:  # same sign, not obv an intermediate root
                done = True
            else:
                print 'warning: need to check all roots!'

        if not done:
            s_coef = h_coef / w
            s_coef[1] -= Rp_z / Rp_x / w
            s_coef[0] += ROf_z - Rp_z / Rp_x * ROf_x

            s_poly = Poly(s_coef)
            roots_u = s_poly.roots()
            roots_t = (roots_u[abs(roots_u.imag) < 1e-7].real / w + ROf_x) / Rp_x
            roots_t_neg = roots_t[roots_t < 0]
            if roots_t_neg.shape[0] > 0:
                t = -INFINITY
                for j in range(roots_t_neg.shape[0]):
                    if roots_t_neg[j] > best_t:
                        t = roots_t_neg[j]
            else:
                print 'warning: choosing positive t!'
                t = roots_t.min()

            for j in range(10):
                u = w * (Rp_x * t - ROf_x)
                y = poly_eval(h_coef, u) / w - fma(Rp_z, t, -ROf_z)
                if fabs(y) < 1e-8: break
                yp = fma(poly_eval(hp_coef, u), Rp_x, -Rp_z)
                t -= y / yp

            # if abs(poly_eval(h_coef, w * (Rp_x * t - ROf_x)) / w - (Rp_z * t - ROf_z)) > 1e-7:
                # print 's vals (u):', s_poly(roots_u)
                # print 'roots_u:', roots_u
                # print 'roots_t:', roots_t
                # print poly_eval_verbose(h_coef, w * (Rp_x * t - ROf_x)) / w - (Rp_z * t - ROf_z)
                # print 'bad root?', t

        ts[i] = t

        if not isfinite(t):
            print 'escape hatch'
            best_t = -INFINITY
            for t0 in np.linspace(0, -1.2, 5):
                t = t0
                for j in range(100):
                    u = w * (Rp_x * t - ROf_x)
                    y = poly_eval(h_coef, u) / w - fma(Rp_z, t, -ROf_z)
                    if fabs(y) < 1e-8: break
                    yp = fma(poly_eval(hp_coef, u), Rp_x, -Rp_z)
                    t -= y / yp

                # print('{: .4f} {: .4f} {: .8f} {: .8f}'.format(t0, t, y, fabs(y)))
                if fabs(y) < 1e-7 and t > best_t + 1e-5 and t < 0:
                    best_t = t

            # if j >= 99:
            #     print "warning iterations exceeded", y + target, target

            # print best_t
            ts[i] = best_t

        # print '{} {}'.format(t0s[i], ts[i])
        assert isfinite(ts[i])
        t = ts[i]
        u = w * (Rp_x * t - ROf_x)
        y = poly_eval(h_coef, u) / w - fma(Rp_z, t, -ROf_z)
        if fabs(y) > 1e-6:
            # print 'big y!', y
            big_ys += 1
        t0s[i] = ts[i]

    # print 'final ts:', ts
    if big_ys > 0:
        print 'big ys:', big_ys

    return ts, ts * rays - ROf[:, np.newaxis]
