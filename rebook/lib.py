import cv2
import numpy as np
import time

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

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('*** timer:%r  %2.2f ms' % \
                (method.__name__, (te - ts) * 1000))
        return result
    return timed
