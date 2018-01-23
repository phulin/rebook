import cv2
import numpy as np
import os
import sys

from numpy import newaxis

import binarize

from algorithm import dominant_char_height
from training import blockwise_coord_descent_mapping, patches

def upscale(path, data_dir, factor):
    assert factor == 2

    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im_h, im_w = im.shape

    bw = binarize.binarize(im, algorithm=binarize.ntirogiannis2014)
    AH = dominant_char_height(bw)

    possible_AHs = np.array([int(d) for d in os.listdir(data_dir) if d.isdigit()])
    size = possible_AHs[np.abs(possible_AHs - AH).argmin()]
    D_T_coupled = np.load(os.path.join(data_dir, str(size), 'dict.npy'))

    W_l = int(size / 3) | 1
    W_h = 2 * W_l
    step = 3

    assert D_T_coupled.shape[1] == W_l * W_l + W_h * W_h
    D_T = D_T_coupled[:, :W_l * W_l]

    K = D_T.shape[0]
    lam = 0.2  # weight of sparsity. TODO: confirm same as training data.

    lo_patches = patches(im, W_l, step)
    M, N, _, _ = lo_patches.shape
    lo_patches_vec = lo_patches.reshape(M * N, W_l * W_l).astype(np.float64)
    means = lo_patches_vec.mean(axis=1)
    X_T = lo_patches_vec - means[:, newaxis]
    t = X_T.shape[0]

    Z_T = np.zeros((t, K), dtype=np.float64)

    last_objective = None
    while True:
        blockwise_coord_descent_mapping(X_T, Z_T, D_T, lam)
        objective = np.square(X_T - Z_T.dot(D_T)).sum()

        print('\ncurrent objective:', objective)
        if last_objective is not None:
            relative_err = abs(last_objective - objective) / last_objective
            print('relative error:', relative_err)
            if relative_err < 1e-4:
                break
        last_objective = objective

    hi_patches_vec = Z_T.dot(D_T_coupled)[:, W_l * W_l:] + means[:, newaxis]
    hi_patches = hi_patches_vec.reshape(M, N, W_h, W_h)

    hi_float = np.zeros((factor * im_h, factor * im_w), dtype=np.float64)
    dest_patches = patches(hi_float, W_h, 2 * step)
    patch_count = np.zeros((factor * im_h, factor * im_w), dtype=int)
    count_patches = patches(patch_count, W_h, 2 * step)
    for i, row in enumerate(hi_patches):
        for j, patch in enumerate(row):
            dest_patches[i, j] += patch
            count_patches[i, j] += 1

    hi_float /= patch_count

    return hi_float

if __name__ == '__main__':
    path = sys.argv[1]
    out = upscale(path, sys.argv[2], 2)
    cv2.imwrite(out, path[:-4] + '_x2' + path[-4:])
