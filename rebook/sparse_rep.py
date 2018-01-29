from __future__ import division, print_function

import numpy as np
from numpy import dot, newaxis
from numpy.linalg import norm, solve
import os
import sys

import lib
from training import print_dict, training_data

def col_square_norm(A):
    return np.einsum('ij, ij->j', A, A)

def row_square_norm(A):
    return np.einsum('ij, ij->i', A, A)

# Optimize B in-place, using Lagrange dual method of:
# Lee et al., Efficient Sparse Coding Algorithms.
# with c=1.
@lib.timeit
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
    print('current D:', D(Lam_vec))
    Lam_vec, _, _ = scipy.optimize.fmin_l_bfgs_b(
        func=lambda x: -D(x),
        bounds=[(0, np.inf) for l in Lam_vec],
        fprime=lambda x: -grad(x),
        x0=Lam_vec
    )
    print('final D:', D(Lam_vec))
    B_T[...] = B(Lam_vec)
    print(B_T)

    return Lam_vec

def solve_cholesky(L, b):
    # solve L L* x = b
    y = solve_triangular(L, b, lower=True)
    return solve_triangular(L.T, y)

@lib.timeit
# @profile
def feature_sign_search_vec(Y_T, X_T, A_T, gamma):
    Y = Y_T.T.copy()
    A = A_T.T.copy()
    X = X_T.T.copy()
    ATA = dot(A_T, A)

    X_T[abs(X_T) < 1e-7] = 0
    active_set = X != 0
    theta = np.sign(X)

    A_T_Y = dot(A_T, Y)

    first_step_2 = True
    last_Is = None

    # shape same as X
    L2_partials = 2 * (dot(ATA, X) - A_T_Y)
    L2_partials_abs = np.abs(L2_partials)

    while True:
        print()
        print('==== STEP 2 ====')

        L2_partials_abs[np.abs(X) >= 1e-7] = 0  # rule out zero elements of X
        Is = L2_partials_abs.argmax(axis=0)  # max for each column

        activate_rows, = np.nonzero(L2_partials_abs.max(axis=0) > gamma)
        index = (Is[activate_rows], activate_rows)
        active_set[index] = True
        theta[index] = -np.sign(L2_partials[index])
        print('mean active:', active_set.sum(axis=0).mean())
        print('activating rows:', activate_rows.shape[0])
        if activate_rows.shape[0] == 0:
            print('WARNING: activating nothing')
        assert last_Is is None or \
            not np.all(last_Is == Is[activate_rows])
        last_Is = Is[activate_rows]

        working_rows = np.arange(X.shape[1]) if first_step_2 else activate_rows
        first_step_2 = False

        while True:
            print('---- STEP 3 ----')
            print('working rows:', working_rows.shape[0])

            Q = A_T_Y[:, working_rows] - gamma / 2 * theta[:, working_rows]
            X_working = X[:, working_rows]
            X_new = X_working.copy()
            Y_working = Y[:, working_rows]
            active_set_working = active_set[:, working_rows]
            for idx, active in enumerate(active_set_working.T):
                active_idxs, = active.nonzero()
                q_hat = Q[active_idxs, idx]
                ATA_hat = ATA[np.ix_(active_idxs, active_idxs)]

                _, x_new_hat, info = scipy.linalg.lapack.dposv(ATA_hat, q_hat)

                if info != 0:
                    x_new_hat = dot(pinv(ATA_hat), q_hat)
                    if np.abs(dot(ATA_hat, x_new_hat) - q_hat).mean() > 0.1:
                        # no good. try null-space zero crossing.
                        active = active_set[:, idx]
                        x_hat = X[active_idxs, idx]
                        theta_hat = theta[active_idxs, idx]
                        u, s, v = np.linalg.svd(ATA_hat)
                        assert s[s.shape[0] - 1] < 1e-7
                        z = v[v.shape[0] - 1]
                        assert np.abs(dot(ATA_hat, z)).sum() < 1e-7
                        # [x_hat + t_i * z]_i = 0
                        # want to reduce theta dot (x + tz) => t * theta dot z
                        # so t should have opposite sign of theta dot z
                        direction = -np.sign(dot(theta_hat, z))
                        null_ts = -x_hat / z
                        null_ts[np.sign(null_ts) != direction] = np.inf
                        null_ts[np.abs(null_ts) < 1e-7] = np.inf
                        first_change = np.abs(null_ts).argmin()
                        x_new_hat = x_hat + null_ts[first_change] * z

                X_new[active_idxs, idx] = x_new_hat

            # sign_changes = np.logical_xor(x_new_hat > 0, x_hat > 0)
            sign_changes = np.logical_and.reduce([
                np.logical_xor(X_new > 0, X_working > 0),
                np.abs(X_working) >= 1e-7,
                # np.abs(X_new) >= 1e-7,
                # np.abs((X_new - X_working) / X_working) >= 1e-9,
            ])

            # (1 - t) * x + t * x_new
            count_sign_changes = sign_changes.sum(axis=0)
            max_sign_changes = count_sign_changes.max()
            has_sign_changes, = np.nonzero(count_sign_changes > 0)
            print('max sign changes:', max_sign_changes)
            print('rows with sign changes:', has_sign_changes.shape[0])

            if max_sign_changes > 0:
                sign_changes = sign_changes[:, has_sign_changes]
                count_sign_changes = count_sign_changes[has_sign_changes]
                Y_sign = Y_working[:, has_sign_changes]
                X_new_sign = X_new[:, has_sign_changes]
                X_sign = X_working[:, has_sign_changes]

                compressed_ts = np.zeros((max_sign_changes, has_sign_changes.shape[0]))
                compressed_mask = np.tile(np.arange(max_sign_changes),
                                          (compressed_ts.shape[1], 1)).T < count_sign_changes
                assert compressed_mask.shape == compressed_ts.shape
                assert compressed_mask.sum() == sign_changes.sum()

                # ts = -x_hat_sign / (x_new_hat_sign - x_hat_sign)
                # NB: only faster to use where= on slow ops like divide.
                all_ts = np.divide(-X_sign, X_new_sign - X_sign, where=sign_changes)

                # transpose necessary to get order right.
                compressed_ts.T[compressed_mask.T] = all_ts.T[sign_changes.T]
                ts = compressed_ts[:, newaxis, :]  # broadcast over components.
                test_X_ts = np.multiply(1 - ts, X_sign, where=compressed_mask[:, newaxis, :]) \
                    + np.multiply(ts, X_new_sign, where=compressed_mask[:, newaxis, :])
                test_X = np.concatenate([test_X_ts, X_new_sign[newaxis, :, :]], axis=0)
                # assert np.sum(test_X[0, X_new_sign != 0] == 0) > 0

                A_X_sign = dot(A, X_sign)
                A_X_new_sign = dot(A, X_new_sign)
                test_A_X_ts = np.multiply(1 - ts, A_X_sign, where=compressed_mask[:, newaxis, :]) \
                    + np.multiply(ts, A_X_new_sign, where=compressed_mask[:, newaxis, :])
                test_A_X = np.concatenate([test_A_X_ts, A_X_new_sign[newaxis, :, :]], axis=0)

                test_mask = np.concatenate([
                    compressed_mask,
                    np.full((1, compressed_mask.shape[1]), True),
                ])
                objectives = np.square(Y_sign - test_A_X, where=test_mask[:, newaxis, :]).sum(axis=1) \
                    + gamma * np.abs(test_X).sum(axis=1)
                objectives[~test_mask] = np.inf
                lowest_objective = objectives.argmin(axis=0)
                best_X = test_X[lowest_objective, :, np.arange(test_X.shape[2])].T
                assert np.all(best_X[:, 0] == test_X[lowest_objective[0], :, 0])

                # # coord_mask = sign_changes[:, 0]
                # coord_mask = active_set_working[:, has_sign_changes][:, 0]
                # shape = X_sign[coord_mask, 0][newaxis, ...].shape
                # debug_array = np.concatenate([
                #     X_sign[coord_mask, 0][newaxis, ...],
                #     X_new_sign[coord_mask, 0][newaxis, ...],
                #     np.full(shape, np.nan),
                #     all_ts[coord_mask, 0][newaxis, ...],
                #     np.full(shape, np.nan),
                #     test_X[:, coord_mask, 0],
                # ])
                # objective_mark = np.zeros(objectives[:, 0].shape)
                # objective_mark[lowest_objective[0]] = -1
                # objective_mark[~test_mask[:, 0]] = np.nan
                # print np.concatenate([
                #     np.concatenate([[np.nan] * 5, compressed_ts[:, 0], [1]])[:, newaxis],
                #     np.full((debug_array.shape[0], 1), np.nan),
                #     debug_array,
                #     np.full((debug_array.shape[0], 1), np.nan),
                #     np.concatenate([[np.nan] * 5, objectives[:, 0]])[:, newaxis],
                #     np.concatenate([[np.nan] * 5, objective_mark])[:, newaxis],
                # ], axis=1)
                # print best_X[coord_mask, 0]

                X_new[:, has_sign_changes] = best_X

            # update x, theta, active set.
            zero_coeffs_mask = np.abs(X_new) < 1e-7
            zero_coeffs = np.nonzero(zero_coeffs_mask)
            X_new[zero_coeffs] = 0
            X[:, working_rows] = X_new
            active_set[:, working_rows] = ~zero_coeffs_mask
            theta[:, working_rows] = np.sign(X_new)

            # objective = np.square(Y - dot(A, X)).sum() + gamma * np.abs(X).sum()
            # print 'CURRENT OBJECTIVE:', objective

            L2_partials_working = 2 * (dot(ATA, X_new) - A_T_Y[:, working_rows])
            f_partials = L2_partials_working + gamma * theta[:, working_rows]
            # only look at max of nonzero coefficients.
            f_partials[zero_coeffs] = 0
            row_highest_nz_partial = np.abs(f_partials).max(axis=0)
            print('highest nonzero partial:', row_highest_nz_partial.max())
            if max_sign_changes == 0 or row_highest_nz_partial.max() < 1e-7:
                break

            working_rows = working_rows[row_highest_nz_partial >= 1e-7]

        np.save('fss_inter.npy', X.T)

        objective = np.square(Y - dot(A, X)).sum() + gamma * np.abs(X).sum()
        print('CURRENT OBJECTIVE:', objective)
        assert objective < 1e11

        zero_coeffs = np.abs(X) < 1e-7
        L2_partials[:, working_rows] = L2_partials_working
        L2_partials_abs[:, working_rows] = np.abs(L2_partials_working)
        highest_zero_partial = L2_partials_abs[zero_coeffs].max()
        print('highest zero partial:', highest_zero_partial)
        if highest_zero_partial <= gamma * 1.01:
            break

    X_T.T = X[:]

@lib.timeit
def feature_sign_search_alternating(X_T, Z_T, D_T, lam):
    feature_sign_search_vec(X_T, Z_T, D_T, lam)
    # feature_sign_search(X_T, Z_T, D_T, lam)
    np.save('fss.npy', Z_T)

    print('optimizing dict.')
    global Lam_last
    Lam_last = optimize_dictionary(X_T, Z_T, D_T, Lam_0=Lam_last)
    np.save('dict.npy', D_T)

def blockwise_coord_descent_mapping(X_T, S_T, B_T, lam):
    alpha = lam / 2.
    K = B_T.shape[0]

    A = B_T.dot(B_T.T)
    np.fill_diagonal(A, 0)
    E = B_T.dot(X_T.T)
    S = S_T.T

    for k in range(K):
        if k % 100 == 0: print(k)
        row = E[k] - A[k].dot(S)
        S[k] = np.maximum(row, alpha) + np.minimum(row, -alpha)

def blockwise_coord_descent_dict(X_T, S_T, B_T, lam):
    K = B_T.shape[0]

    G = S_T.T.dot(S_T)
    np.fill_diagonal(G, 0)
    W = X_T.T.dot(S_T)

    for k in range(K):
        row = W[:, k] - B_T.T.dot(G[:, k])
        B_T[k] = row / norm(row)

@lib.timeit
def blockwise_coord_descent(X_T, S_T, B_T, lam):
    blockwise_coord_descent_mapping(X_T, S_T, B_T, lam)
    np.save('fss.npy', S_T)

    blockwise_coord_descent_dict(X_T, S_T, B_T, lam)
    np.save('dict.npy', B_T)

def test_train():
    W_l = 5  # window size
    W_h = 2 * W_l

    font_size = 56

    K = 512  # Dictionary size
    lam = 0.1  # weight of sparsity

    if os.path.isfile('training.npy'):
        X_T = np.load('training.npy')
    else:
        X_T = training_data("/Library/Fonts/Microsoft/Constantia.ttf",
                            font_size, W_l, W_h)
        np.save('training.npy', X_T)

    t = X_T.shape[0]

    if os.path.isfile('fss.npy'):
        Z_T = np.load('fss.npy')
    else:
        Z_T = np.zeros((t, K), dtype=np.float64)

    if os.path.isfile('dict.npy'):
        D_T = np.load('dict.npy')
    else:
        D_T = np.random.normal(size=(K, W_l * W_l + W_h * W_h)).astype(np.float64)
        # D_T = X_T[np.random.choice(X_T.shape[0], size=K, replace=False)]
        D_T /= norm(D_T, axis=1)[:, newaxis]
        np.save('dict.npy', D_T)

    print('shapes:', X_T.shape, Z_T.shape, D_T.shape)

    global Lam_last
    Lam_last = None
    last_objective = None
    for i in range(100000):
        print('\n==== ITERATION', i, '====')
        # feature_sign_search_alternating(X_T, Z_T, D_T, lam)
        blockwise_coord_descent(X_T, Z_T, D_T, lam)

        print_dict('lo_dict.png'.format(i), D_T[:, :W_l * W_l])
        print_dict('hi_dict.png'.format(i), D_T[:, W_l * W_l:])

        highest = Z_T.argmax()
        weight = Z_T.flat[highest]
        patch_X, patch_D = np.unravel_index(highest, Z_T.shape)
        print('highest weight:', weight)
        print(weight * D_T[patch_D, :W_l * W_l].reshape(W_l, W_l))
        print(weight * D_T[patch_D, W_l * W_l:].reshape(W_h, W_h))
        print(X_T[patch_X, :W_l * W_l].reshape(W_l, W_l))
        print(X_T[patch_X, W_l * W_l:].reshape(W_h, W_h))
        print(dot(Z_T[patch_X], D_T)[:W_l * W_l])

        diff = (X_T - dot(Z_T, D_T)).reshape(-1)
        objective = dot(diff, diff).sum() + lam * abs(Z_T).sum()
        print('\nTOTAL OBJECTIVE VALUE:', objective)

        if last_objective is not None:
            relative_err = abs(last_objective - objective) / last_objective
            print('relative error:', relative_err)
            if relative_err < 1e-4:
                break
        last_objective = objective

def train(dest, font_path, sizes):
    for size in sizes:
        W_l = int(size / 3) | 1
        W_h = 2 * W_l

        K = 64  # Dictionary size
        lam = 0.2  # weight of sparsity

        dest_dir = os.path.join(dest, str(size))
        if not os.path.isdir(dest_dir):
            print('making directory', dest_dir)
            os.makedirs(dest_dir)

        training_file = os.path.join(dest_dir, 'training.npy')
        dict_file = os.path.join(dest_dir, 'dict.npy')
        mapping_file = os.path.join(dest_dir, 'mapping.npy')

        if os.path.isfile(training_file):
            X_T = np.load(training_file)
        else:
            X_T = training_data(font_path, size * 2, W_l, W_h)
            np.save(training_file, X_T)

        t = X_T.shape[0]

        if os.path.isfile(mapping_file):
            Z_T = np.load(mapping_file)
        else:
            Z_T = np.zeros((t, K), dtype=np.float64)

        if os.path.isfile(dict_file):
            D_T = np.load(dict_file)
        else:
            D_T = np.random.normal(size=(K, W_l * W_l + W_h * W_h)).astype(np.float64)
            D_T /= norm(D_T, axis=1)[:, newaxis]
            np.save(dict_file, D_T)

        last_objective = None
        for i in range(100000):
            print('\n==== ITERATION', i, '====')
            blockwise_coord_descent(X_T, Z_T, D_T, lam)
            np.save(mapping_file, Z_T)
            np.save(dict_file, D_T)

            print_dict(os.path.join(dest_dir, 'lo_dict.png'), D_T[:, :W_l * W_l])
            print_dict(os.path.join(dest_dir, 'hi_dict.png'), D_T[:, W_l * W_l:])

            objective = np.square(X_T - dot(Z_T, D_T)).sum() + lam * abs(Z_T).sum()
            print('\nTOTAL OBJECTIVE VALUE:', objective)

            if last_objective is not None:
                relative_err = abs(last_objective - objective) / last_objective
                print('relative error:', relative_err)
                if relative_err < 1e-4:
                    break
            last_objective = objective

if __name__ == '__main__':
    lib.debug = True
    # lib.debug_prefix = 'training/'
    np.set_printoptions(precision=3, linewidth=200)
    # these sizes should correspond to AH in scanned stuff.
    sizes = [15, 18, 20, 22, 26, 30]
    train(sys.argv[1], "/Library/Fonts/Microsoft/Constantia.ttf", sizes)
