# coding=utf-8

import cv2
import freetype
import math
import numpy as np
# import numpy.ma as ma
import os
import scipy.optimize
import sys

from numpy import dot, newaxis
from numpy.linalg import norm, solve

import lib

from feature_sign import feature_sign_search

chars = u'abcdefghijklmnopqrstuvwxyz' + \
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
    u'1234567890.\'",§¶()-;:'

LO_SIZE = 28
HI_SIZE = 2 * LO_SIZE

def create_mosaic(face, size):
    face.set_pixel_sizes(0, size)
    rendered = []
    for c in chars:
        face.load_char(c)
        height = face.glyph.bitmap.rows
        bitmap = np.array(face.glyph.bitmap.buffer, dtype=np.uint8).reshape(height, -1).copy()
        rendered.append(bitmap)

    padded = []
    for g in rendered:
        padding_x = 2 * size - g.shape[0]
        padding_y = 2 * size - g.shape[1]
        padding = ((padding_x / 2, padding_x - padding_x / 2),
                   (padding_y / 2, padding_y - padding_y / 2))
        padded.append(np.pad(g, padding, 'constant'))

    return np.concatenate(padded, axis=1)

# step: distance between patches
def patches(a, size, step=1):
    patch_count = (
        (a.shape[0] - size) / step + 1,
        (a.shape[1] - size) / step + 1,
    )
    return np.lib.stride_tricks.as_strided(
        a, patch_count + (size, size),
        (step * a.strides[0], step * a.strides[1]) + a.strides
    )

def col_square_norm(A):
    return np.einsum('ij, ij->j', A, A)

def row_square_norm(A):
    return np.einsum('ij, ij->i', A, A)

# Optimize B in-place, using Lagrange dual method of:
# Lee et al., Efficient Sparse Coding Algorithms.
# with c=1.
def optimize_dictionary(X_T, S_T, B_T, Lam_0=None):
    SST = dot(S_T.T, S_T)
    XST = dot(X_T.T, S_T)
    XST_T = XST.T.copy()
    XTX = dot(X_T, X_T.T)
    XSTTXST = dot(XST_T, XST)

    def B(Lam_vec):
        Lam = np.diag(Lam_vec)
        return solve(SST + Lam, XST_T)

    def D(Lam_vec):
        Lam = np.diag(Lam_vec)
        return np.trace(XTX) - np.trace(Lam) \
            - np.trace(XST.dot(solve(SST + Lam, XST_T)))

    def grad(Lam_vec):
        Lam = np.diag(Lam_vec)
        return row_square_norm(solve(SST + Lam, XST_T)) - 1

    def hessian(Lam, inv_SST_Lam):
        return -2 * inv_SST_Lam \
            * (inv_SST_Lam.dot(XSTTXST).dot(inv_SST_Lam))

    # last_B_T = None
    Lam_vec = np.ones(S_T.shape[1]) if Lam_0 is None else Lam_0.copy()
    print 'current D:', D(Lam_vec)
    Lam_vec, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=lambda x: -D(x),
        bounds=[(0, np.inf) for l in Lam_vec],
        fprime=lambda x: -grad(x),
        x0=Lam_vec
    )
    print 'final D:', D(Lam_vec)
    B_T[...] = B(Lam_vec)
    print B_T

    return Lam_vec

def print_dict(filename, D_T):
    K, W_sq = D_T.shape
    W = int(math.sqrt(W_sq))
    assert W_sq == W ** 2

    D_T_s = D_T - np.percentile(D_T, 5)
    ratio = 255 / np.percentile(D_T_s, 95)
    patches = lib.clip_u8(ratio * D_T_s.reshape(K, W, W))

    sqrtK = int(math.ceil(math.sqrt(K)))
    padding = ((0, sqrtK ** 2 - K), (1, 1), (1, 1))
    patches_padded = np.pad(patches, padding, 'constant', constant_values=127)
    dict_square = patches_padded.reshape(sqrtK, sqrtK, W + 2, W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtK * (W + 2), sqrtK * (W + 2))

    lib.debug_imwrite(filename, dict_square)

def feature_sign_search_vec(Y_T, X_T, A_T, gamma):
    Y = Y_T.T.copy()
    A = A_T.T.copy()
    X = X_T.T
    ATA = dot(A_T, A)

    X_T[abs(X_T) < 1e-7] = 0
    active_set = X != 0
    theta = np.sign(X)

    A_T_Y = dot(A_T, Y)

    first_step_2 = True

    while True:
        print
        print '==== STEP 2 ===='
        # STEP 2
        # shape same as X
        L2_partials = 2 * (dot(ATA, X) - A_T_Y)
        L2_partials_abs = np.abs(L2_partials)

        L2_partials_abs[np.abs(X) >= 1e-7] = 0  # rule out zero elements of X
        Is = L2_partials_abs.argmax(axis=0)  # max for each column

        activate_rows = L2_partials_abs.max(axis=0) > gamma
        index = (Is[activate_rows], np.nonzero(activate_rows)[0])
        active_set[index] = True
        theta[index] = -np.sign(L2_partials[index])
        print 'mean active:', active_set.sum(axis=0).mean()
        print 'activating rows:', activate_rows.sum()

        working_rows = np.ones(activate_rows.shape, dtype=bool) \
            if first_step_2 else activate_rows
        first_step_2 = False

        while True:
            print '---- STEP 3 ----'

            Q = A_T_Y[:, working_rows] - gamma / 2 * theta[:, working_rows]
            X_working = X[:, working_rows]
            X_new = X_working.copy()
            Y_working = Y[:, working_rows]
            active_set_working = active_set[:, working_rows]
            for idx, active in enumerate(active_set_working.T):
                ATA_hat = ATA[np.ix_(active, active)]
                X_new[active, idx] = solve(ATA_hat, Q[active, idx])

            # null_rows = np.abs(dot(ATA, X_new) - Q).sum(axis=0) \
            #     >= 1e-3 * norm(Q, axis=0)

            # for i in np.nonzero(null_rows)[0]:
            #     # no good. try null-space zero crossing
            #     print 'null row!', i
            #     active = active_set[:, i]
            #     x_hat = X[active, i]
            #     theta_hat = theta[active, i]
            #     ATA_hat = ATA[np.ix_(active, active)]
            #     u, s, v = np.linalg.svd(ATA_hat)
            #     assert s[s.shape[0] - 1] < 1e-7
            #     z = v[v.shape[0] - 1]
            #     assert np.abs(dot(ATA_hat, z)).sum() < 1e-7
            #     # print 'z:', z
            #     # [x_hat + t_i * z]_i = 0
            #     # want to reduce theta dot (x + tz) => t * theta dot z
            #     # so t should have opposite sign of theta dot z
            #     direction = -np.sign(dot(theta_hat, z))
            #     null_ts = -x_hat / z
            #     null_ts[np.sign(null_ts) != direction] = np.inf
            #     null_ts[np.abs(null_ts) < 1e-7] = np.inf
            #     first_change = np.abs(null_ts).argmin()
            #     X_new[active, i] = x_hat + null_ts[first_change] * z

            # sign_changes = np.logical_xor(x_new_hat > 0, x_hat > 0)
            sign_changes = np.logical_and(
                np.logical_xor(X_new > 0, X_working > 0),
                np.abs(X_working) >= 1e-7  # don't select zero coefficients of x_hat.
            )

            # (1 - t) * x + t * x_new
            count_sign_changes = sign_changes.sum(axis=0)
            max_sign_changes = count_sign_changes.max()
            has_sign_changes = count_sign_changes > 0
            print 'max sign changes:', max_sign_changes
            print 'rows with sign changes:', has_sign_changes.sum()

            if max_sign_changes > 0:
                sign_changes = sign_changes[:, has_sign_changes]
                count_sign_changes = count_sign_changes[has_sign_changes]
                Y_sign = Y_working[:, has_sign_changes]
                X_new_sign = X_new[:, has_sign_changes]
                X_sign = X_working[:, has_sign_changes]
                X_new_minus_X = X_new_sign - X_sign

                compressed_ts = np.zeros((max_sign_changes, has_sign_changes.sum()))
                compressed_mask = np.tile(np.arange(max_sign_changes),
                                          (compressed_ts.shape[1], 1)).T < count_sign_changes
                assert compressed_mask.shape == compressed_ts.shape
                assert compressed_mask.sum() == sign_changes.sum()

                # ts = -x_hat_sign / (x_new_hat_sign - x_hat_sign)
                # NB: only faster to use where= on slow ops like divide.
                ts = np.divide(-X_sign, X_new_minus_X, where=sign_changes)
                compressed_ts.T[compressed_mask.T] = ts.T[sign_changes.T]
                test_X = np.concatenate([
                    X_sign + compressed_ts[:, newaxis, :] * X_new_minus_X[newaxis, :, :],
                    X_new_sign[newaxis, :, :]
                ], axis=0)
                # assert np.sum(test_X[0, X_new_sign != 0] == 0) > 0

                diffs = Y_sign[:, newaxis, :] - dot(A, test_X)
                objectives = np.einsum('ijk,ijk->jk', diffs, diffs) \
                    + gamma * np.abs(test_X).sum(axis=1)
                lowest_objective = objectives.argmin(axis=0)
                best_X = test_X[lowest_objective, :, np.arange(test_X.shape[2])].T
                assert np.all(best_X[:, 0] == test_X[lowest_objective[0], :, 0])

                X_new[:, has_sign_changes] = best_X

            # update x, theta, active set.
            X[:, working_rows] = X_new
            zero_coeffs = np.abs(X) < 1e-7
            X[zero_coeffs] = 0
            active_set[zero_coeffs] = False
            theta = np.sign(X)

            objective = np.square(Y - dot(A, X)).sum() + gamma * np.abs(X).sum()
            print 'CURRENT OBJECTIVE:', objective

            # diff = y - dot(A, x)
            # current_objective = dot(diff, diff) + gamma * abs(x).sum()
            # # print 'x:', x[x != 0]
            # # print 'CURRENT OBJECTIVE:', dot(diff, diff), '+', gamma * abs(x).sum()
            # assert current_objective < last_objective + 1e-7
            # last_objective = current_objective

            L2_partials = 2 * (dot(ATA, X) - A_T_Y)
            f_partials = L2_partials + gamma * theta
            print 'still nonoptimal:', np.sum(np.abs(f_partials[~zero_coeffs]) >= 1e-7)
            if np.all(np.abs(f_partials[~zero_coeffs]) < 1e-7):
                break

        print 'highest zero partial:', np.abs(L2_partials[zero_coeffs]).max()
        if np.all(np.abs(L2_partials[zero_coeffs]) <= gamma):
            break

def make_dicts(argv):
    face = freetype.Face("/Library/Fonts/Microsoft/Constantia.ttf")

    hi_res = create_mosaic(face, HI_SIZE)
    cv2.imwrite('hi.png', hi_res)

    blurred = hi_res  # cv2.blur(hi_res, (5, 5))
    lo_res = cv2.resize(blurred, (0, 0), None, 0.5, 0.5,
                        interpolation=cv2.INTER_AREA)
    cv2.imwrite('lo.png', lo_res)

    W_l = 5  # window size
    W_h = 2 * W_l
    # make sure we're on edges (in hi-res reference)
    counts = cv2.boxFilter(hi_res.clip(0, 1), -1, (W_h, W_h), normalize=False)
    edge_patches = np.logical_and(counts > 4 * W_l, counts < W_h * W_h - 4 * W_l)

    # these two arrays should correspond
    patch_centers = edge_patches[W_l - 1:-W_l:4, W_l - 1:-W_l:4]
    lo_patches = patches(lo_res, W_l, 2)[patch_centers]
    hi_patches = patches(hi_res, W_h, 4)[patch_centers]
    t = lo_patches.shape[0]
    print 'patches:', t

    print lo_patches[2000]
    print hi_patches[2000]

    lo_patches_vec = lo_patches.reshape(t, W_l * W_l).astype(np.float64)
    print_dict('lo_sq.png', lo_patches_vec)
    lo_patches_vec -= lo_patches_vec.mean(axis=1)[:, newaxis]
    print lo_patches_vec[2000]
    hi_patches_vec = hi_patches.reshape(t, W_h * W_h).astype(np.float64)
    print_dict('hi_sq.png', hi_patches_vec)
    hi_patches_vec -= hi_patches_vec.mean(axis=1)[:, newaxis]
    print hi_patches_vec[2000]

    coupled_patches = np.concatenate([
        lo_patches_vec / W_l,
        hi_patches_vec / W_h
    ], axis=1)
    # Z = argmin_Z( ||X-DZ||_2^2 ) + lam ||Z||_1
    # D: (W_l*W_l, K); Z: (K, t); X: (W_l*W_l, t)

    K = 1024  # Dictionary size
    lam = 0.1  # weight of sparsity
    X_T = coupled_patches
    Z_T = np.zeros((t, K), dtype=np.float64)

    # D_T = X_T[np.random.choice(X_T.shape[0], size=K, replace=False)]
    D_T = np.random.normal(size=(K, W_l * W_l + W_h * W_h)).astype(np.float64)
    D_T /= norm(D_T, axis=1)[:, newaxis]
    print X_T.shape, Z_T.shape, D_T.shape

    Lam_last = None
    D_T_last = None
    for i in range(100000):
        print '\n==== ITERATION', i, '===='
        if i == 0 and os.path.isfile('fss.npy'):
            Z_T = np.load('fss.npy')
        else:
            feature_sign_search_vec(X_T, Z_T, D_T, lam)
            np.save('fss.npy', Z_T)

        diff = (X_T - dot(Z_T, D_T)).reshape(-1)
        objective = dot(diff, diff).sum() + lam * abs(Z_T).sum()
        print '\nTOTAL OBJECTIVE VALUE:', objective

        print 'optimizing dict.'
        Lam_last = optimize_dictionary(X_T, Z_T, D_T, Lam_0=Lam_last)
        np.save('dict.npy', D_T)

        if D_T_last is not None:
            relative_err = abs(D_T - D_T_last).mean() / abs(D_T_last).mean()
            print 'relative error:', relative_err
            if relative_err < 1e-4:
                break
        D_T_last = D_T.copy()

        print_dict('lo_dict{}.png'.format(i), D_T[:, :W_l * W_l])
        print_dict('hi_dict{}.png'.format(i), D_T[:, W_l * W_l:])

        highest = Z_T.argmax()
        weight = Z_T.flat[highest]
        patch_X, patch_D = np.unravel_index(highest, Z_T.shape)
        print 'highest weight:', weight
        print weight * D_T[patch_D, :W_l * W_l].reshape(W_l, W_l)
        print weight * D_T[patch_D, W_l * W_l:].reshape(W_h, W_h)
        print lo_patches[patch_X]
        print hi_patches[patch_X]
        print dot(Z_T[patch_X], D_T)[:W_l * W_l]

        diff = (X_T - dot(Z_T, D_T)).reshape(-1)
        objective = dot(diff, diff).sum() + lam * abs(Z_T).sum()
        print '\nTOTAL OBJECTIVE VALUE:', objective

if __name__ == '__main__':
    lib.debug = True
    lib.debug_prefix = 'training/'
    make_dicts(sys.argv)
