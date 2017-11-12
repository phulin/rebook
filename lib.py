import cv2
import numpy as np

debug = False
def debug_imwrite(path, im):
    if debug:
        cv2.imwrite(path, im)

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
