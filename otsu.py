import cv2
import sys
import os
import re
from multiprocessing.pool import ThreadPool

from lib import gradient, vert_close, text_contours, yan, hsl_gray

indir = sys.argv[1]
outdir = sys.argv[2]

pool = ThreadPool(2)

files = filter(lambda f: re.search('.(png|jpg|tif)$', f), os.listdir(indir))

def crop(im):
    im_w, im_h = len(im[0]), len(im)

    grad = gradient(im)
    closed = vert_close(grad)
    cv2.imwrite("grad.png", grad)
    cv2.imwrite("vert_close.png", closed)

    good_contours, bad_contours = text_contours(closed)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    crop_x0, crop_y0, crop_x1, crop_y1 = im_w, im_h, 0, 0
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        crop_x0 = min(x, crop_x0)
        crop_y0 = min(y, crop_y0)
        crop_x1 = max(x + w, crop_x1)
        crop_y1 = max(y + h, crop_y1)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for c in bad_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 4)

    cv2.imwrite("debug.png", debug)

    crop_x0 = int(max(0, crop_x0 - .01 * im_h))
    crop_y0 = int(max(0, crop_y0 - .01 * im_h))
    crop_x1 = int(min(im_w, crop_x1 + .01 * im_h))
    crop_y1 = int(min(im_h, crop_y1 + .01 * im_h))

    return im[crop_y0:crop_y1, crop_x0:crop_x1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
def go(fn):
    print fn
    inpath = os.path.join(indir, fn)
    outpath = os.path.join(outdir, fn)
    if os.path.isfile(outpath):
        print'skipping', outpath
        return
    original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_UNCHANGED)
    # img = cv2.GaussianBlur(img, (5, 5), 0)
    img = cv2.resize(original, (0, 0), None, 1.5, 1.5)
    if len(img.shape) > 2:
        print 'color'
        s, l = hsl_gray(img)
        print fn, 'clahe'
        s, l = clahe.apply(s), clahe.apply(l)
        cv2.imwrite("saturation.png", s)
        cv2.imwrite("luminance.png", l)
        print fn, 'threshold'
        bw = yan(l) # & yan(l, T=35)
    # elif (original + 1 < 2).all():  # black and white
    #     print 'bw'
    #     _, bw = cv2.threshold(img, 100, 255, cv2.THRESH_BINARY)
    else:
        print 'grayscale'
        print fn, 'clahe'
        img = clahe.apply(img)
        cv2.imwrite('clahe.png', img)
        print fn, 'threshold'
        bw = yan(img, T=35)
    print fn, 'thresholded'
    cv2.imwrite('thresholded.png', bw)
    bw = crop(bw)
    cv2.imwrite(outpath, bw)

# for fn in files:
    # go(fn)
pool.map(go, files)
