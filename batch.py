import cv2
import sys
import os
import re
from multiprocessing.pool import Pool

from lib import gradient, text_contours, binarize, roth, skew_angle, safe_rotate

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

    return ((crop_x0, crop_y0), (crop_x1, crop_y1))

def crop(im, bw, split=True):
    im_w, im_h = len(im[0]), len(im)

    grad = gradient(bw)
    cv2.imwrite("grad.png", grad)
    space_width = (im_h / 50) | 1
    line_height = (im_h / 100) | 1
    box = cv2.getStructuringElement(cv2.MORPH_RECT, (space_width, line_height))
    closed = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, box)
    cv2.imwrite("closed.png", closed)

    open_size = (im_h / 800) | 1
    cross = cv2.getStructuringElement(cv2.MORPH_CROSS, (open_size, open_size))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, cross)
    cv2.imwrite("opened.png", opened)

    good_contours, bad_contours = text_contours(opened, bw)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for c in bad_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 4)

    if split and im_w > im_h:  # two pages
        cv2.imwrite("debug.png", debug)
        contour_sets = split_contours(good_contours)
    else:
        contour_sets = [good_contours]

    return [crop_to_contours(im, cs) for cs in contour_sets]

def go(inpath, outpath):
    print inpath
    original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_UNCHANGED)
    im_h, im_w = original.shape
    bw = binarize(original, algorithm=roth, resize=1.0)
    cv2.imwrite('thresholded.png', bw)
    crops = crop(original, bw)
    for idx, c in enumerate(crops):
        (x0, y0), (x1, y1) = c
        if x1 > x0 and y1 > y0:
            bw_cropped = bw[y0:y1, x0:x1]
            orig_cropped = original[y0:y1, x0:x1]
            angle = skew_angle(bw_cropped)
            rotated = safe_rotate(orig_cropped, angle)
            rotated_bw = binarize(rotated, algorithm=roth, resize=1.0)
            new_crop = crop(rotated, rotated_bw, split=False)[0]
            (x0r, y0r), (x1r, y1r) = new_crop
            outimg = rotated_bw[y0r:y1r, x0r:x1r]
            cv2.imwrite('{}_{}{}'.format(outpath[:-4], idx, '.tif'), outimg)

concurrent = False

if __name__ == '__main__':
    indir = sys.argv[1]
    outdir = sys.argv[2]

    files = filter(lambda f: re.search('.(png|jpg|tif)$', f),
                   os.listdir(indir))
    files.sort()
    infiles = [os.path.join(indir, f) for f in files]
    outfiles = [os.path.join(outdir, f) for f in files]

    im = cv2.imread(infiles[0], cv2.CV_LOAD_IMAGE_UNCHANGED)
    im_h, im_w = im.shape
    # image height should be about 10 inches. round to 100
    dpi = int(round(im_h / 1000.0) * 100)

    if concurrent:
        pool = Pool(2)
        pool.map(go, files)
    else:
        map(go, infiles, outfiles)
