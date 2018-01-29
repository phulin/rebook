# coding=utf-8

from __future__ import division, print_function

import cv2
import freetype
import numpy as np

from numpy import newaxis

import lib

# from feature_sign import feature_sign_search

chars = 'abcdefghijklmnopqrstuvwxyz' + \
    'ABCDEFGHIJKLMNOPQRSTUVWXYZ' + \
    '1234567890.\'",§¶()-;:'

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
        padding = ((padding_x // 2, padding_x - padding_x // 2),
                   (padding_y // 2, padding_y - padding_y // 2))
        padded.append(np.pad(g, padding, 'constant'))

    return np.concatenate(padded, axis=1)

# step: distance between patches
# looks for image in first two axes
def patches(a, size, step=1):
    patch_count = (
        (a.shape[0] - size) // step + 1,
        (a.shape[1] - size) // step + 1,
    )
    return np.lib.stride_tricks.as_strided(
        a, patch_count + (size, size) + a.shape[2:],
        (step * a.strides[0], step * a.strides[1]) + a.strides
    )

def print_dict(filename, D_T):
    K, W_sq = D_T.shape
    W = int(np.sqrt(W_sq))
    assert W_sq == W ** 2

    D_T_s = D_T - np.percentile(D_T, 5)
    ratio = 255 / np.percentile(D_T_s, 95)
    patches = lib.clip_u8(ratio * D_T_s.reshape(K, W, W))

    sqrtK = int(np.ceil(np.sqrt(K)))
    padding = ((0, sqrtK ** 2 - K), (1, 1), (1, 1))
    patches_padded = np.pad(patches, padding, 'constant', constant_values=127)
    dict_square = patches_padded.reshape(sqrtK, sqrtK, W + 2, W + 2) \
        .transpose(0, 2, 1, 3).reshape(sqrtK * (W + 2), sqrtK * (W + 2))

    lib.debug_imwrite(filename, dict_square)

def training_data(font_path, font_size, W_l, W_h):
    face = freetype.Face(font_path)

    hi_res = create_mosaic(face, font_size)
    cv2.imwrite('hi.png', hi_res)

    blur_size = (W_l // 2) | 1
    blurred = cv2.blur(hi_res, (blur_size, blur_size))
    lo_res = cv2.resize(blurred, (0, 0), None, 0.5, 0.5,
                        interpolation=cv2.INTER_AREA)
    cv2.imwrite('lo.png', lo_res)

    # make sure we're on edges (in hi-res reference)
    counts = cv2.boxFilter(hi_res.clip(0, 1), -1, (W_h, W_h), normalize=False)
    edge_patches = np.logical_and(counts > 4 * W_l, counts < W_h * W_h - 4 * W_l)

    # these two arrays should correspond
    patch_centers = edge_patches[W_l - 1:-W_l:4, W_l - 1:-W_l:4]
    lo_patches = patches(lo_res, W_l, 2)[patch_centers]
    hi_patches = patches(hi_res, W_h, 4)[patch_centers]
    t = lo_patches.shape[0]
    print('patches:', t)

    print(lo_patches[100])
    print(hi_patches[100])

    lo_patches_vec = lo_patches.reshape(t, W_l * W_l).astype(np.float64)
    print_dict('lo_sq.png', lo_patches_vec)
    lo_patches_vec -= lo_patches_vec.mean(axis=1)[:, newaxis]
    print(lo_patches_vec[100])
    hi_patches_vec = hi_patches.reshape(t, W_h * W_h).astype(np.float64)
    print_dict('hi_sq.png', hi_patches_vec)
    hi_patches_vec -= hi_patches_vec.mean(axis=1)[:, newaxis]
    print(hi_patches_vec[100])

    coupled_patches = np.concatenate([
        lo_patches_vec / W_l,
        hi_patches_vec / W_h
    ], axis=1)
    # Z = argmin_Z( ||X-DZ||_2^2 ) + lam ||Z||_1
    # D: (W_l*W_l, K); Z: (K, t); X: (W_l*W_l, t)

    return coupled_patches
