import cv2
import sys
import os
import re
import numpy as np
from multiprocessing.pool import ThreadPool
from skimage.filters import threshold_sauvola

indir = sys.argv[1]
outdir = sys.argv[2]

pool = ThreadPool(2)

files = filter(lambda f: re.search('.(png|jpg|tif)$', f), os.listdir(indir))

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def gradient(im):
    space_width = len(im) / 100 * 2 + 1
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (space_width, 1))
    im = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, horiz)

def vert_close(im):
    space_width = len(im) / 200 * 2 + 1
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, space_width))
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, vert)

def sauvola(im):
    thresh = threshold_sauvola(im, window_size=len(im) / 300)
    booleans = im > (thresh * 1.0)
    ints = booleans.astype(np.uint8) * 255
    return ints

def crop(im):
    im_w, im_h = len(im[0]), len(im)
    min_feature_size = im_h / 300

    grad = gradient(im)
    copy = vert_close(grad).copy()
    cv2.rectangle(copy, (0, 0), (im_w, im_h), 255, 3)
    cv2.imwrite("grad.png", grad)
    cv2.imwrite("vert_close.png", copy)
    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    contours, [hierarchy] = \
        cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest holes
    image_area = im_w * im_h
    good_holes = []
    i = 0
    while i >= 0:
        j = hierarchy[i][2]
        while j >= 0:
            c = contours[j]
            x, y, w, h = cv2.boundingRect(c)
            if w * h > image_area * 0.25:
                good_holes.append(j)
            j = hierarchy[j][0]
        i = hierarchy[i][0]

    crop_x0, crop_y0, crop_x1, crop_y1 = im_w, im_h, 0, 0
    for hole in good_holes:
        x, y, w, h = cv2.boundingRect(contours[hole])
        print "hole:", x, y, w, h

        i = hierarchy[hole][2]
        while i >= 0:
            c = contours[i]
            x, y, w, h = cv2.boundingRect(c)
            if len(c) > 10 \
                    and h < 3 * w \
                    and w > min_feature_size \
                    and h > min_feature_size \
                    and x > 0.02 * im_w \
                    and x + w < 0.98 * im_w \
                    and y > 0.02 * im_h \
                    and y + h < 0.98 * im_h:
                crop_x0 = min(x, crop_x0)
                crop_y0 = min(y, crop_y0)
                crop_x1 = max(x + w, crop_x1)
                crop_y1 = max(y + h, crop_y1)
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 4)
            i = hierarchy[i][0]

    cv2.imwrite("debug.png", debug)

    crop_x0 = int(max(0, crop_x0 - .01 * im_h))
    crop_y0 = int(max(0, crop_y0 - .01 * im_h))
    crop_x1 = int(min(im_w, crop_x1 + .01 * im_h))
    crop_y1 = int(min(im_h, crop_y1 + .01 * im_h))

    return im[crop_y0:crop_y1, crop_x0:crop_x1]

def go(fn):
    print fn
    inpath = os.path.join(indir, fn)
    img = cv2.imread(inpath, 0)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(img, (0, 0), None, 1.5, 1.5)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    print fn, 'clahe'
    img = clahe.apply(img)
    print fn, 'sauvola'
    bw = sauvola(img)
    print fn, 'thresholded'
    bw = crop(bw)
    # ret, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    outpath = os.path.join(outdir, fn)
    cv2.imwrite(outpath, bw)

for fn in files:
    go(fn)
# pool.map(go, files)
