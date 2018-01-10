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

    D_T_s = D_T + D_T.min(axis=1)[:, newaxis]
    ratio = 255 / np.percentile(D_T_s, 99)
    patches = lib.clip_u8(ratio * D_T_s[:, :W * W].reshape(K, W, W))

    sqrtK = int(math.ceil(math.sqrt(K)))
    padding = ((0, sqrtK ** 2 - K), (1, 1), (1, 1))
    patches_padded = np.pad(patches, padding, 'constant')
    dict_square = patches_padded.reshape(sqrtK, sqrtK, W + 2, W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtK * (W + 2), sqrtK * (W + 2))

    lib.debug_imwrite(filename, dict_square)

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
        hi_patches_vec / (W_h)
    ], axis=1)
    # Z = argmin_Z( ||X-DZ||_2^2 ) + lam ||Z||_1
    # D: (W_l*W_l, K); Z: (K, t); X: (W_l*W_l, t)

    K = 1024  # Dictionary size
    lam = 0.1  # weight of sparsity
    D_T = np.random.normal(size=(K, W_l * W_l + W_h * W_h)).astype(np.float64)
    D_T /= norm(D_T, axis=1)[:, newaxis]

    X_T = coupled_patches
    Z_T = np.zeros((t, K), dtype=np.float64)
    print X_T.shape, Z_T.shape, D_T.shape

    Lam_last = None
    D_T_last = None
    for i in range(50):
        print '\n==== ITERATION', i, '===='
        if i == 0 and os.path.isfile('fss.npy'):
            Z_T = np.load('fss.npy')
        else:
            feature_sign_search(X_T, Z_T, D_T, lam)
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
            if relative_err < 1e-3:
                break
        D_T_last = D_T.copy()

        print_dict('lo_dict{}.png'.format(i), D_T[:, :W_l * W_l])
        print_dict('hi_dict{}.png'.format(i), D_T[:, W_l * W_l:])

        diff = (X_T - dot(Z_T, D_T)).reshape(-1)
        objective = dot(diff, diff).sum() + lam * abs(Z_T).sum()
        print '\nTOTAL OBJECTIVE VALUE:', objective

if __name__ == '__main__':
    make_dicts(sys.argv)
