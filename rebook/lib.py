from __future__ import division, print_function

import cv2
import numpy as np
import os
import os.path
import rawpy
import time

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

debug = False
debug_prefix = []
def debug_imwrite(filename, im):
    if not debug: return False

    if debug_prefix:
        directory = os.path.join(*debug_prefix)
        if not os.path.isdir(directory):
            os.makedirs(directory)
    else:
        directory = '.'

    return cv2.imwrite(os.path.join(directory, filename), im)

def imread(path):
    if path.endswith('.dng'):
        with rawpy.imread(path) as raw:
            result = raw.postprocess()
    else:
        result = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    return result

def normalize_u8(im):
    im_max = im.max()
    im_min = im.min()
    alpha = 255.0 / (im_max - im_min)
    return (alpha * (im - im_min)).astype(np.uint8)

def clip_u8(im):
    return im.clip(0, 255).astype(np.uint8)

def bool_to_u8(bools):
    return -bools.astype(np.uint8, copy=False)

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

def round_point(p):
    try:
        return tuple(np.round(np.atleast_1d(p)).astype(int))
    except:
        return (0, 0)

def draw_line(debug, p1, p2, color=GREEN, thickness=2):
    cv2.line(debug, round_point(p1), round_point(p2), color, thickness)

def draw_circle(debug, p, radius=2, color=GREEN, thickness=cv2.FILLED):
    cv2.circle(debug, round_point(p), radius, color, thickness)
