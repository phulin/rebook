# coding=utf-8

import cv2
import freetype
import math
import numpy as np
# import numpy.ma as ma
import os
import sys

from numpy import dot, newaxis
from numpy.linalg import inv, norm

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

# Optimize B in-place, using Lagrange dual method of:
# Lee et al., Efficient Spare Coding Algorithms.
# with c=1.
def optimize_dictionary(X_T, S_T, B_T):
    SST = dot(S_T.T, S_T)
    XST = dot(X_T.T, S_T)
    XST_T = XST.T.copy()
    XTX = dot(X_T, X_T.T)
    XSTTXST = dot(XST_T, XST)

    def B(Lam, inv_SST_Lam):
        return dot(inv_SST_Lam, XST_T)

    def D(Lam, inv_SST_Lam):
        return np.trace(XTX) - np.trace(Lam) \
            - np.trace(XST.dot(inv_SST_Lam).dot(XST_T))

    def grad(Lam, inv_SST_Lam):
        return np.sum(dot(XST, inv_SST_Lam) ** 2, axis=0) - 1

    def hessian(Lam, inv_SST_Lam):
        return -2 * inv_SST_Lam \
            * (inv_SST_Lam.dot(XSTTXST).dot(inv_SST_Lam))

    last_B_T = None
    Lam_vec = np.ones(S_T.shape[1])
    for i in range(50):
        gamma = 0.5  # relaxed newton's method
        Lam = np.diag(Lam_vec)
        inv_SST_Lam = inv(SST + Lam)

        H = hessian(Lam, inv_SST_Lam)
        G = grad(Lam, inv_SST_Lam)
        H_inv_G = np.linalg.solve(H, G)
        Lam_vec = Lam_vec - gamma * H_inv_G

        B_T[...] = B(Lam, inv_SST_Lam)
        print 'B_T:', B_T
        if last_B_T is not None and np.abs(B_T - last_B_T).mean() < 1e-4 * np.abs(last_B_T).mean():
            break
        last_B_T = B_T.copy()

def make_dicts(argv):
    face = freetype.Face("/Library/Fonts/Microsoft/Constantia.ttf")

    hi_res = create_mosaic(face, HI_SIZE)
    cv2.imwrite('hi.png', hi_res)

    blurred = hi_res  # cv2.blur(hi_res, (5, 5))
    lo_res = cv2.resize(blurred, (0, 0), None, 0.5, 0.5,
                        interpolation=cv2.INTER_AREA)
    cv2.imwrite('lo.png', lo_res)

    W = 5  # window size
    # make sure we're on edges (in hi-res reference)
    counts = cv2.boxFilter(hi_res.clip(0, 1), -1, (2 * W, 2 * W), normalize=False)
    edge_patches = np.logical_and(counts > 4 * W, counts < 4 * W * W - 4 * W)

    # these two arrays should correspond
    patch_centers = edge_patches[W - 1:-W:4, W - 1:-W:4]
    lo_patches = patches(lo_res, W, 2)[patch_centers]
    hi_patches = patches(hi_res, 2 * W, 4)[patch_centers]
    t = lo_patches.shape[0]
    print 'patches:', t

    sqrtt = int(math.ceil(math.sqrt(t)))
    padding = ((0, sqrtt ** 2 - t), (1, 1), (1, 1))
    hi_dict_padded = np.pad(hi_patches, padding, 'constant')
    hi_dict_square = hi_dict_padded.reshape(sqrtt, sqrtt, 2 * W + 2, 2 * W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtt * (2 * W + 2), sqrtt * (2 * W + 2))
    cv2.imwrite('hi_sq.png', hi_dict_square)

    print lo_patches[2000]
    print hi_patches[2000]

    K = 1024  # Dictionary size
    lam = 0.1  # weight of sparsity
    D_T = np.random.normal(size=(K, 5 * W * W)).astype(np.float64)
    D_T /= norm(D_T, axis=1)[:, newaxis]

    lo_patches_vec = lo_patches.reshape(t, W * W).astype(np.float64)
    lo_patches_vec -= lo_patches_vec.mean(axis=1)[:, newaxis]
    hi_patches_vec = hi_patches.reshape(t, 4 * W * W).astype(np.float64)
    hi_patches_vec -= hi_patches_vec.mean(axis=1)[:, newaxis]
    coupled_patches = np.concatenate([
        lo_patches_vec / W,
        hi_patches_vec / (2 * W)
    ], axis=1)
    # Z = argmin_Z( ||X-DZ||_2^2 ) + lam ||Z||_1
    # D: (W*W, K); Z: (K, t); X: (W*W, t)

    X_T = coupled_patches
    Z_T = np.zeros((t, K), dtype=np.float64)
    print X_T.shape, Z_T.shape, D_T.shape

    D_T_last = None
    for i in range(50):
        print '==== ITERATION', i, '===='
        if i == 0 and os.path.isfile('fss.npy'):
            Z_T = np.load('fss.npy')
        else:
            feature_sign_search(X_T, Z_T, D_T, lam)
            np.save('fss.npy', Z_T)
        print 'optimizing dict.'
        optimize_dictionary(X_T, Z_T, D_T)
        np.save('dict.npy', D_T)

        if D_T_last is not None:
            relative_err = abs(D_T - D_T_last).mean() / abs(D_T_last).mean()
            print 'relative error:', relative_err
            if relative_err < 1e-3:
                break
        D_T_last = D_T.copy()

    print D_T.shape
    print np.percentile(D_T, 30)
    print np.percentile(D_T, 50)
    print np.percentile(D_T, 70)
    print np.percentile(D_T, 90)
    print np.percentile(D_T, 99)
    ratio = 255 / np.percentile(D_T, 99)
    lo_dict = lib.clip_u8(ratio * D_T[:, :W * W].reshape(K, W, W))
    hi_dict = lib.clip_u8(ratio * D_T[:, W * W:].reshape(K, 2 * W, 2 * W))

    sqrtK = int(math.ceil(math.sqrt(K)))
    padding = ((0, sqrtK ** 2 - K), (1, 1), (1, 1))
    lo_dict_padded = np.pad(lo_dict, padding, 'constant')
    hi_dict_padded = np.pad(hi_dict, padding, 'constant')
    lo_dict_square = lo_dict_padded.reshape(sqrtK, sqrtK, W + 2, W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtK * (W + 2), sqrtK * (W + 2))
    hi_dict_square = hi_dict_padded.reshape(sqrtK, sqrtK, 2 * W + 2, 2 * W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtK * (2 * W + 2), sqrtK * (2 * W + 2))

    cv2.imwrite('lo_dict.png', lo_dict_square)
    cv2.imwrite('hi_dict.png', hi_dict_square)

if __name__ == '__main__':
    make_dicts(sys.argv)
