import cv2
import numpy as np
from skimage.filters import threshold_sauvola, threshold_niblack

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

def sauvola(im, window_factor=200, k=0.2, thresh_factor=1.0):
    thresh = threshold_sauvola(im, window_size=len(im) / window_factor * 2 + 1)
    booleans = im > (thresh * thresh_factor)
    ints = booleans.astype(np.uint8) * 255
    return ints

def niblack(im):
    thresh = threshold_niblack(im, window_size=len(im) / 200 * 2 + 1)
    booleans = im > (thresh * 1.0)
    ints = booleans.astype(np.uint8) * 255
    return ints

def kittler(im):
    h, g = np.histogram(im.ravel(), 256, [0, 256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p = c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - \
        p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    _, thresh = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)
    return thresh

def roth(im, s=51, t=0.8):
    im_h, im_w = im.shape
    means = cv2.blur(im, (s, s))
    booleans = im > means * t
    ints = booleans.astype(np.uint8) * 255
    return ints

# s = stroke width
def kamel(im, s=None, T=25):
    im_h, im_w = im.shape
    if s is None or s <= 0:
        s = im_h / 200
    size = 2 * s + 1
    means = cv2.blur(im, (size, size), borderType=cv2.BORDER_REFLECT_101)
    padded = np.pad(means, (s, s), 'edge')
    im_plus_T = im.astype(np.int64) + T
    im_plus_T = im_plus_T.clip(min=0, max=255).astype(np.uint8)
    L1 = padded[0:im_h, 0:im_w] < im_plus_T
    L2 = padded[0:im_h, s:im_w + s] < im_plus_T
    L3 = padded[0:im_h, 2 * s:im_w + 2 * s] < im_plus_T
    L4 = padded[s:im_h + s, 2 * s:im_w + 2 * s] < im_plus_T
    L5 = padded[2 * s:im_h + 2 * s, 2 * s:im_w + 2 * s] < im_plus_T
    L6 = padded[2 * s:im_h + 2 * s, s:im_w + s] < im_plus_T
    L7 = padded[2 * s:im_h + 2 * s, 0:im_w] < im_plus_T
    L0 = padded[s:im_h + s, 0:im_w] < im_plus_T
    b = (L0 & L1 & L4 & L5) | (L1 & L2 & L5 & L6) | \
        (L2 & L3 & L6 & L7) | (L3 & L4 & L7 & L0)

    return b.astype(np.uint8) * 255

def otsu(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def text_contours(im):
    im_w, im_h = len(im[0]), len(im)
    min_feature_size = im_h / 300

    copy = im.copy()
    cv2.rectangle(copy, (0, 0), (im_w, im_h), 255, 3)
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

    good_contours, bad_contours = [], []
    for hole in good_holes:
        x, y, w, h = cv2.boundingRect(contours[hole])
        print "hole:", x, y, w, h

        i = hierarchy[hole][2]
        while i >= 0:
            c = contours[i]
            x, y, w, h = cv2.boundingRect(c)
            print 'contour:', x, y, w, h
            if len(c) > 10 \
                    and h < 3 * w \
                    and w > min_feature_size \
                    and h > min_feature_size \
                    and x > 0.02 * im_w \
                    and x + w < 0.98 * im_w \
                    and y > 0.02 * im_h \
                    and y + h < 0.98 * im_h:
                good_contours.append(c)
            else:
                bad_contours.append(c)
            i = hierarchy[i][0]

    return good_contours, bad_contours
