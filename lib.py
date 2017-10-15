import cv2
import numpy as np

debug = True
def debug_imwrite(path, im):
    if debug:
        cv2.imwrite(path, im)

def normalize_u8(im):
    im_max = im.max()
    alpha = 255 / im_max
    beta = im.min() * im_max / 255
    return cv2.convertScaleAbs(im, alpha=alpha, beta=beta, dtype=np.uint8)

def clip_u8(im):
    return im.clip(0, 255).astype(np.uint8)

def bool_to_u8(im):
    return im.astype(np.uint8) - 1
