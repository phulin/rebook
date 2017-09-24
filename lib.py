import cv2
import numpy as np
from skimage.filters import threshold_sauvola

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
    thresh = threshold_sauvola(im, window_size=len(im) / 400 * 2 + 1)
    booleans = im > (thresh * 1.0)
    ints = booleans.astype(np.uint8) * 255
    return ints

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

