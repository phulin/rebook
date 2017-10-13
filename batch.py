import cv2
import sys
import os
import re
from multiprocessing.pool import Pool

from lib import gradient, vert_close, text_contours, binarize, roth, \
    skew_angle, safe_rotate

indir = sys.argv[1]
outdir = sys.argv[2]

files = filter(lambda f: re.search('.(png|jpg|tif)$', f), os.listdir(indir))

def split_contours(contours):
    # Maximize horizontal separation
    # sorted by starting x value, ascending).
    x_contours = [(cv2.boundingRect(c)[0], c) for c in contours]
    sorted_contours = sorted(x_contours, key=lambda (x, c): x)

    # Greedy algorithm. Maximize L bound of R minus R bound of L.
    current_r = 0
    quantity = 0
    argmax = -1
    for idx in range(len(contours) - 1):
        x1, _, w, _ = cv2.boundingRect(sorted_contours[idx][1])
        current_r = max(current_r, x1 + w)
        x2, _ = sorted_contours[idx + 1]
        if x2 - current_r > quantity:
            quantity = x2 - current_r
            argmax = idx

    print 'split:', argmax, 'out of', len(contours), '@', current_r

    sorted_contours = [c for x, c in sorted_contours]
    return sorted_contours[:argmax + 1], sorted_contours[argmax + 1:]

def crop_to_contours(im, contour_set):
    im_w, im_h = len(im[0]), len(im)

    crop_x0, crop_y0, crop_x1, crop_y1 = im_w, im_h, 0, 0
    for c in contour_set:
        x, y, w, h = cv2.boundingRect(c)
        crop_x0 = min(x, crop_x0)
        crop_y0 = min(y, crop_y0)
        crop_x1 = max(x + w, crop_x1)
        crop_y1 = max(y + h, crop_y1)

    crop_x0 = int(max(0, crop_x0 - .01 * im_h))
    crop_y0 = int(max(0, crop_y0 - .01 * im_h))
    crop_x1 = int(min(im_w, crop_x1 + .01 * im_h))
    crop_y1 = int(min(im_h, crop_y1 + .01 * im_h))

    return im[crop_y0:crop_y1, crop_x0:crop_x1]

def crop(im):
    im_w, im_h = len(im[0]), len(im)

    grad = gradient(im)
    # closed = vert_close(grad)
    cv2.imwrite("grad.png", grad)
    space_width = (im_h / 50) | 1
    line_height = (im_h / 100) | 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                        (space_width, line_height))
    closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, ellipse)
    cv2.imwrite("closed.png", closed)

    good_contours, bad_contours = text_contours(closed)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for c in bad_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 4)

    if im_w > im_h:  # two pages
        cv2.imwrite("debug.png", debug)
        contour_sets = split_contours(good_contours)
    else:
        contour_sets = [good_contours]

    return [crop_to_contours(im, cs) for cs in contour_sets]

def go(fn):
    print fn
    inpath = os.path.join(indir, fn)
    outpath = os.path.join(outdir, fn)
    if os.path.isfile(outpath):
        print'skipping', outpath
        return
    original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_UNCHANGED)
    im_h, im_w = original.shape
    bw = binarize(original, algorithm=roth, resize=1.0)
    cv2.imwrite('thresholded.png', bw)
    out = crop(bw)
    for idx, outimg in enumerate(out):
        if len(outimg) > 0:
            angle = skew_angle(outimg)
            outimg = safe_rotate(outimg, angle)
            outimg = crop(outimg)[0]
            cv2.imwrite('{}_{}{}'.format(outpath[:-4], idx, '.tif'), outimg)

for fn in files:
    go(fn)

# pool = Pool(4)
# pool.map(go, files)
