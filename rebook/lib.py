from __future__ import division, print_function

import cv2
import numpy as np
import time

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

debug = False
debug_prefix = ''
def debug_imwrite(path, im):
    if debug:
        cv2.imwrite(debug_prefix + path, im)

def normalize_u8(im):
    im_max = im.max()
    im_min = im.min()
    alpha = 255.0 / (im_max - im_min)
    return (alpha * (im - im_min)).astype(np.uint8)

def clip_u8(im):
    return im.clip(0, 255).astype(np.uint8)

def bool_to_u8(im):
    return im.astype(np.uint8) - 1

def is_bw(im):
    return (im + 1 < 2).all()

def int_tuple(a):
    return tuple(np.round(a).astype(int))

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('*** timer:%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result
    return timed

def mean_std(im, W):
    s = W // 2
    N = W * W

    padded = np.pad(im, (s, s), 'reflect')
    sum1, sum2 = cv2.integral2(padded, sdepth=cv2.CV_32F)

    S1 = sum1[W:, W:] - sum1[W:, :-W] - sum1[:-W, W:] + sum1[:-W, :-W]
    S2 = sum2[W:, W:] - sum2[W:, :-W] - sum2[:-W, W:] + sum2[:-W, :-W]

    means = S1 / N

    variances = S2 / N - means * means
    stds = np.sqrt(variances.clip(0, None))

    return means, stds
