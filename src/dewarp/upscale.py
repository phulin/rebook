import cv2
import numpy as np
import os
import sys

from numpy import newaxis

import binarize
import training

from algorithm import dominant_char_height
from training import patches

def upscale(path, data_dir, factor):
    assert factor == 2

    im = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    im_h, im_w = im.shape

    bw = binarize.binarize(im, algorithm=binarize.ntirogiannis2014)
    AH = dominant_char_height(bw)
    print('AH =', AH)

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
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, struct)
    gradient_means, _ = binarize.mean_std(gradient, W_l)
    patch_gradient = gradient_means[W_l // 2:-W_l // 2 + 1:step,
                                    W_l // 2:-W_l // 2 + 1:step]
    assert patch_gradient.shape == (lo_patches.shape[0], lo_patches.shape[1])
    patch_mask = patch_gradient > np.percentile(patch_gradient, 50)
    good_patches = np.nonzero(patch_mask)
    cv2.imwrite('spp.png', -patch_mask.astype(np.uint8))

    lo_patches_vec = lo_patches[good_patches].reshape(-1, W_l * W_l).astype(np.float64)
    means = lo_patches_vec.mean(axis=1)
    X_T = lo_patches_vec - means[:, newaxis]
    t = X_T.shape[0]

    Z_T = np.zeros((t, K), dtype=np.float64)

    last_objective = None
    while True:
        training.feature_sign_search_vec(X_T, Z_T, D_T, lam)
        objective = np.square(X_T - Z_T.dot(D_T)).sum()

        print('\ncurrent objective:', objective)
        if last_objective is not None:
            relative_err = abs(last_objective - objective) / last_objective
            print('relative error:', relative_err)
            if relative_err < 1e-4:
                break
        last_objective = objective

    hi_patches = Z_T.dot(D_T_coupled)[:, W_l * W_l:] + means[:, newaxis]

    hi_float = np.zeros((factor * im_h, factor * im_w), dtype=np.float64)
    dest_patches = patches(hi_float, W_h, 2 * step)
    patch_count = np.zeros((factor * im_h, factor * im_w), dtype=int)
    count_patches = patches(patch_count, W_h, 2 * step)
    for i, patch in enumerate(hi_patches):
        dest_patches[good_patches[i]] += patch
        count_patches[good_patches[i]] += 1
    np.divide(hi_float, patch_count, hi_float, where=patch_count > 0)

    hi_lanczos = cv2.resize(im, (0, 0), None, 2., 2.,
                            interpolation=cv2.INTER_LANCZOS4)

    return np.where(patch_count > 0, hi_float, hi_lanczos).clip(0, 255).astype(np.uint8)

if __name__ == '__main__':
    path = sys.argv[1]
    out = upscale(path, sys.argv[2], 2)
    cv2.imwrite(out, path[:-4] + '_x2' + path[-4:])
