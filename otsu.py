import cv2
import sys
import os
import re
import multiprocessing as mp
import numpy as np
from multiprocessing.pool import ThreadPool
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

indir = sys.argv[1]
outdir = sys.argv[2]

pool = ThreadPool(mp.cpu_count())

files = filter(lambda f: re.search('.(png|jpg|tif)$', f), os.listdir(indir))

def sauvola(im):
    thresh = threshold_sauvola(im, window_size=15)
    booleans = im > (thresh * 1.1)
    ints = booleans.astype(np.uint8) * 255
    return ints

def go(fn):
    print fn
    inpath = os.path.join(indir, fn)
    img = cv2.imread(inpath, 0)
    img = cv2.resize(img, (0, 0), None, 1.5, 1.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = clahe.apply(img)
    # img = sauvola(img)
    img = img[300:5700, 800:8900]
    ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    outpath = os.path.join(outdir, fn)
    cv2.imwrite(outpath, th)


pool.map(go, files)
