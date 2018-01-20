# cython: boundscheck=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: cdivision=True

from __future__ import division
import numpy as np
cimport numpy as np

import sys

from numpy import dot, einsum
from numpy.linalg import norm, solve

cimport cython

cdef np.ndarray[np.float64_t, ndim=1] \
row_square_norm(np.ndarray[np.float64_t, ndim=2] A):
    return einsum('ij, ij->i', A, A)

# Optimize Z in-place.
def feature_sign_search(np.ndarray[np.float64_t, ndim=2] Y_T,
                        np.ndarray[np.float64_t, ndim=2] X_T,
                        np.ndarray[np.float64_t, ndim=2] A_T,
                        double gamma):
    ATA = dot(A_T, A_T.T)

    for idx in range(X_T.shape[0]):
        if idx % 250 == 0:
            print '\nX[{: 5d}]'.format(idx),
            sys.stdout.flush()
        elif idx % 10 == 0:
            sys.stdout.write('.')
            sys.stdout.flush()

        print 'X[{: 5d}]'.format(idx)
        feature_sign_search_single(Y_T[idx], X_T[idx], gamma, A_T, ATA)

    print

def feature_sign_search_single(np.ndarray[np.float64_t, ndim=1] y,
                               np.ndarray[np.float64_t, ndim=1] x,
                               double gamma,
                               np.ndarray[np.float64_t, ndim=2] A_T,
                               np.ndarray[np.float64_t, ndim=2] ATA):
    cdef np.ndarray[np.float64_t, ndim=2] ATA_hat, A_hat_T, test_xs_hat, \
        u, v, diffs
    cdef np.ndarray[np.float64_t, ndim=1] theta, A_T_y, theta_hat, x_hat, \
        x_new_hat, ts, null_ts, best_x, diff, L2_partials, L2_partials_abs, \
        f_partials, x_hat_sign, x_new_hat_sign, s
    cdef np.ndarray[np.int_t, ndim=1] active
    cdef np.ndarray active_set, sign_changes, zero_coeffs
    cdef int lowest_objective, last_selected, i
    cdef double last_objective, direction

    x[abs(x) < 1e-7] = 0
    print 'beginning x:', x[x != 0]
    active_set = x != 0
    theta = np.sign(x)

    A_T_y = dot(A_T, y)

    last_selected = -1
    last_objective = np.inf

    while True:
        print
        print '==== STEP 2 ===='
        L2_partials = 2 * (dot(ATA, x) - A_T_y)
        L2_partials_abs = np.abs(L2_partials)

        L2_partials_abs[active_set] = 0  # max zero elements of x
        i = L2_partials_abs.argmax()
        if L2_partials_abs[i] > gamma:
            # print 'selected', i, 'at dL2/dxi =', L2_partials_abs[i]
            assert last_selected != i
            active_set[i] = True
            theta[i] = -np.sign(L2_partials[i])
            last_selected = i

        while True:
            print '---- STEP 3 ----'
            active, = np.nonzero(active_set)
            print 'active_set:', active

            ATA_hat = ATA[np.ix_(active, active)]
            A_hat_T = A_T[active]
            A_hat_T_y = A_T_y[active]
            theta_hat = theta[active]
            x_hat = x[active]
            print 'x_hat:', x_hat

            q = A_hat_T_y - gamma * theta_hat / 2
            x_new_hat = solve(ATA_hat, q)
            if np.abs(dot(ATA_hat, x_new_hat) - q).sum() >= 1e-7 * abs(q).mean():
                # still no good. try null-space zero crossing
                print 'trying null vec'
                u, s, v = np.linalg.svd(ATA_hat)
                assert s[s.shape[0] - 1] < 1e-7
                z = v[v.shape[0] - 1]
                assert np.abs(dot(ATA_hat, z)).sum() < 1e-7
                # print 'z:', z
                # [x_hat + t_i * z]_i = 0
                # want to reduce theta dot (x + tz) => t * theta dot z
                # so t should have opposite sign of theta dot z
                direction = -np.sign(dot(theta_hat, z))
                null_ts = -x_hat / z
                null_ts[np.sign(null_ts) != direction] = np.inf
                null_ts[np.abs(null_ts) < 1e-7] = np.inf
                first_change = np.abs(null_ts).argmin()
                x_new_hat = x_hat + null_ts[first_change] * z

            print 'x_new:', x_new_hat

            # sign_changes = np.logical_xor(x_new_hat > 0, x_hat > 0)
            sign_changes = np.logical_and(
                np.logical_xor(x_new_hat > 0, x_hat > 0),
                np.abs(x_hat) >= 1e-7  # don't select zero coefficients of x_hat.
            )
            x_hat_sign = x_hat[sign_changes]
            x_new_hat_sign = x_new_hat[sign_changes]
            ts = -x_hat_sign / (x_new_hat_sign - x_hat_sign)
            if ts.shape[0] == 0 or np.abs(ts - 1).min() > 1e-7:
                ts = np.concatenate([ts, [1]])
            print 'ts:', ts

            # (1 - t) * x + t * x_new
            test_xs_hat = x_hat + np.outer(ts, x_new_hat - x_hat)
            test_A_xs = np.outer(1 - ts, dot(x_hat, A_hat_T)) \
                + np.outer(ts, dot(x_new_hat, A_hat_T))
            print test_xs_hat

            objectives = np.square(y - test_A_xs).sum(axis=1) \
                + gamma * np.abs(test_xs_hat).sum(axis=1)
            print 'objectives:', objectives
            print 'old objective:', np.square(y - dot(x_hat, A_hat_T)).sum() + gamma * np.abs(x_hat).sum()

            lowest_objective = objectives.argmin()
            best_x = test_xs_hat[lowest_objective]
            print best_x

            # update x, theta, active set.
            zero_coeffs, = np.nonzero(np.abs(best_x) < 1e-9)
            print 'deactivating:', active[zero_coeffs]
            best_x[zero_coeffs] = 0
            x[active] = best_x
            theta[active] = np.sign(best_x)
            active_set[zero_coeffs] = False

            diff = y - dot(x, A_T)
            current_objective = dot(diff, diff) + gamma * abs(x).sum()
            print 'x:', x[active]
            print 'last objective:', last_objective
            print 'CURRENT OBJECTIVE:', current_objective, '=', \
                dot(diff, diff), '+', gamma * abs(x).sum()
            assert current_objective < last_objective + 1e-7
            last_objective = current_objective
            print 'last objective:', last_objective

            zero_coeffs, = np.nonzero(np.abs(x) < 1e-9)
            L2_partials = 2 * (dot(ATA, x) - A_T_y)
            f_partials = L2_partials + gamma * theta
            if np.all(np.abs(f_partials[np.abs(x) >= 1e-9]) < 1e-7):
                break

        print 'highest zero partial:', abs(L2_partials[zero_coeffs]).max()
        if np.all(np.abs(L2_partials[zero_coeffs]) <= gamma):
            break
