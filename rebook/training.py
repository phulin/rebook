# coding=utf-8

import cv2
import freetype
import numpy as np
import sys

from math import sqrt
from numpy.linalg import inv, norm

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
def optimize_dictionary(X, S, B):
    SST = S.dot(S.T)
    XST = X.dot(S.T)
    XTX = X.T.dot(X)
    XSTTXST = XST.T.dot(XST)

    def D(Lam, inv_SST_Lam):
        return np.trace(XTX - XST.dot(inv_SST_Lam).dot(XST.T) - Lam)

    def grad(Lam):
        return np.sum(XST.dot(SST + Lam) ** 2, axis=0) - 1

    def hessian(Lam, inv_SST_Lam):
        return -2 * inv_SST_Lam \
            * (inv_SST_Lam.dot(XSTTXST).dot(inv_SST_Lam))

    Lam_vec = np.ones(S.shape[0])
    for i in range(10):
        gamma = 0.8  # relaxed newton's method
        Lam = np.diag(Lam_vec)
        inv_SST_Lam = inv(SST + Lam)

        H = hessian(Lam, inv_SST_Lam)
        G = grad(Lam)
        Lam_vec = Lam_vec - gamma * inv(H).dot(G)

        print 'D:', D(Lam, inv_SST_Lam)

def make_dicts(argv):
    face = freetype.Face("/Library/Fonts/Microsoft/Constantia.ttf")

    hi_res = create_mosaic(face, HI_SIZE)
    cv2.imwrite('hi.png', hi_res)

    blurred = hi_res  # cv2.blur(hi_res, (5, 5))
    lo_res = cv2.resize(blurred, (0, 0), None, 0.5, 0.5,
                        interpolation=cv2.INTER_AREA)
    cv2.imwrite('lo.png', lo_res)

    W = 7  # window size
    # make sure we're on edges (in hi-res reference)
    counts = cv2.boxFilter(hi_res.clip(0, 1), -1, (2 * W, 2 * W), normalize=False)
    edge_patches = np.logical_and(counts > 4 * W, counts < 4 * W * W - 4 * W)

    # these two arrays should correspond
    patch_centers = edge_patches[W - 1:-W:2, W - 1:-W:2]
    lo_patches = patches(lo_res, W, 1)[patch_centers]
    hi_patches = patches(hi_res, 2 * W, 2)[patch_centers]
    t = lo_patches.shape[0]
    sqrtt = int(sqrt(t))
    padding = ((0, (sqrtt + 1) * (sqrtt + 1) - t), (0, 0), (0, 0))
    lo_patches = np.pad(lo_patches, padding, 'constant')
    hi_patches = np.pad(hi_patches, padding, 'constant')
    print lo_patches.shape, hi_patches.shape
    print lo_patches[1000]
    print hi_patches[1000]
    print 'sqrtt:', sqrtt

    K = 1024  # Dictionary size
    lam = 0.1  # weight of sparsity
    t = lo_patches.shape[0]
    D = np.random.normal(size=(W * W, K))
    D = D / norm(D, axis=0)

    lo_patches_vec = lo_patches.reshape(t, W * W)
    hi_patches_vec = hi_patches.reshape(t, 4 * W * W)
    # Z = argmin_Z( ||X-DZ||_2^2 ) + lam ||Z||_1
    # D: (W*W, K); Z: (K, t); X: (W*W, t)

    X = lo_patches_vec
    Z = np.zeros((K, t), dtype=np.float64)
    feature_sign_search(X, D, lam, solution=Z)
    optimize_dictionary(X, Z, D)

if __name__ == '__main__':
    make_dicts(sys.argv)
