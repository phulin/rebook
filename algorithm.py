import cv2
import math
import numpy as np

from lib import debug_imwrite
from binarize import binarize

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def gradient(im):
    return cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)

def hsl_gray(im):
    assert len(im.shape) == 3
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    _, l, s = cv2.split(hls)
    return s, l

def text_contours(im, original):
    im_h, im_w = im.shape
    min_feature_size = im_h / 300

    copy = im.copy()
    cv2.rectangle(copy, (0, 0), (im_w, im_h), 255, 3)
    _, contours, [hierarchy] = \
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
        # print "hole:", x, y, w, h

        i = hierarchy[hole][2]
        while i >= 0:
            c = contours[i]
            x, y, w, h = cv2.boundingRect(c)
            # print 'mean:', orig_slice.mean(), \
            # 'horiz stddev:', orig_slice.mean(axis=0).std()
            # print 'contour:', x, y, w, h
            if len(c) > 10 \
                    and h < 2 * w \
                    and w > min_feature_size \
                    and h > min_feature_size:
                mask = np.zeros((h, w), dtype=np.uint8)
                cv2.drawContours(mask, contours, i, 255,
                                 thickness=cv2.FILLED,
                                 offset=(-x, -y))
                mask_filled = np.count_nonzero(mask)

                orig_slice = cv2.bitwise_not(original[y:y + h, x:x + w])
                orig_filled = np.count_nonzero(mask & orig_slice)

                filled_ratio = orig_filled / float(mask_filled)
                if filled_ratio > 0.1:
                    good_contours.append(c)
            else:
                bad_contours.append(c)
            i = hierarchy[i][0]

    mask = np.zeros(im.shape, dtype=np.uint8)
    for i in range(len(good_contours)):
        cv2.drawContours(mask, good_contours, i, 255,
                         thickness=cv2.FILLED)
    debug_imwrite('text_contours.png', mask)

    return good_contours, bad_contours

# and x > 0.02 * im_w \
# and x + w < 0.98 * im_w \
# and y > 0.02 * im_h \
# and y + h < 0.98 * im_h:

def _line_contours(im):
    im_h, _ = im.shape
    bw = ~binarize(im)
    debug_imwrite('bw.png', bw)

    space_width = (im_h / 70) | 1
    # line_height = (im_h / 300) | 1

    # struct = np.zeros((line_height, space_width), dtype=np.uint8)
    # cv2.line(struct, (0, 0), (space_width - 1, line_height - 1), 1)
    # cv2.line(struct, (space_width - 1, 0), (0, line_height - 1), 1)
    # bw1 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, struct)

    # circle = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
    #                                    (line_height, line_height))
    # bw2 = cv2.morphologyEx(bw1, cv2.MORPH_CLOSE, circle)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (space_width, 1))
    bw2 = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, horiz)

    line_contours, _ = text_contours(bw2, im)

    return bw, line_contours

def skew_angle(im):
    small_bw, line_contours = _line_contours(im)

    lines = cv2.cvtColor(small_bw, cv2.COLOR_GRAY2RGB)
    alphas = []
    for c in line_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 * h:
            vx, vy, x1, y1 = cv2.fitLine(c, cv2.DIST_L2, 0, 0.01, 0.01)
            cv2.line(lines,
                     (x1 - vx * 10000, y1 - vy * 10000),
                     (x1 + vx * 10000, y1 + vy * 10000),
                     (255, 0, 0), thickness=3)
            alphas.append(math.atan2(vy, vx))

    debug_imwrite('lines.png', lines)
    return np.median(alphas)

def lu_dewarp(im):
    # morphological operators
    morph_a = [
        np.array([1] + [0] * (2 * i), dtype=np.uint8).reshape(2 * i + 1, 1) \
        for i in range(9)
    ]
    morph_d = [a.T for a in morph_a]
    morph_c = [
        np.array([0] * (2 * i) + [1], dtype=np.uint8).reshape(2 * i + 1, 1) \
        for i in range(9)
    ]
    # morph_b = [c.T for c in morph_c]

    im_inv = im ^ 255
    bdyt = np.zeros(im.shape, dtype=np.uint8) - 1
    for struct in morph_c + morph_d:  # ++ morph_b
        bdyt &= cv2.erode(im_inv, struct)

    debug_imwrite("bdyt.png", bdyt)
    return bdyt

    for struct in morph_c + morph_d:
        bdyt &= im_inv ^ cv2.erode(im_inv, struct)

def top_contours(contours, hierarchy):
    i = 0
    result = []
    while i >= 0:
        result.append(contours[i])
        i = hierarchy[i][0]

    return result

def dominant_char_height(im):
    _, contours, [hierarchy] = cv2.findContours(im, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)

    char_heights = [
        cv2.boundingRect(c)[3] for c in top_contours(contours, hierarchy)
    ]

    hist, _ = np.histogram(char_heights, 256, [0, 256])
    print hist
    return np.argmax(hist)

def dewarp_text(im):
    # Goal-Oriented Rectification

    AH = dominant_char_height(im)
    print 'AH =', AH

    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (int(AH * 0.6) | 1, 1))
    rls = cv2.morphologyEx(im ^ 255, cv2.MORPH_CLOSE, horiz)
    debug_imwrite('rls.png', rls)

    _, contours, [hierarchy] = cv2.findContours(rls, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
    words = top_contours(contours, hierarchy)
    word_boxes = [tuple([word] + list(cv2.boundingRect(word))) for word in words]
    word_boxes = filter(
        lambda (_, x, y, w, h): h < 4 * AH and h > AH / 3 and w > AH / 3,
        word_boxes
    )
    word_boxes.sort(key=lambda (word, x, y, w, h): x)

    lines = []
    for word_box in word_boxes:
        _, x1, y1, w1, h1 = word_box
        # print "word:", x1, y1, w1, h1
        candidates = []
        for l in lines:
            _, x2, y2, w2, h2 = l[-1]
            if x2 < x1 and x1 < x2 + w2 + 6 * AH and y2 <= y1 + h1 and y1 <= y2 + h2:
                # print "  candidate:", x2, y2, w2, h2
                candidates.append((x1 - x2 - w2, l))

        if candidates:
            candidates.sort()
            _, line = candidates[-1]
            _, x, y, w, h = line[-1]
            line.append(word_box)
            # print "  selected:", x, y, w, h
        else:
            lines.append([word_box])

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for _, x, y, w, h in word_boxes:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for l in lines:
        min_x1 = min((x for _, x, y, w, h in l))
        max_x2 = max((x + w for _, x, y, w, h in l))
        min_y1 = min((y for _, x, y, w, h in l))
        max_y2 = max((y + h for _, x, y, w, h in l))
        cv2.rectangle(debug, (min_x1, min_y1), (max_x2, max_y2), (255, 0, 0), 2)
    debug_imwrite('debug.png', debug)

    return im

def safe_rotate(im, angle):
    debug_imwrite('prerotated.png', im)
    im_h, im_w = im.shape
    if abs(angle) > math.pi / 4:
        print "warning: too much rotation"
        return im

    im_h_new = im_w * abs(math.sin(angle)) + im_h * math.cos(angle)
    im_w_new = im_h * abs(math.sin(angle)) + im_w * math.cos(angle)

    pad_h = int(math.ceil((im_h_new - im_h) / 2))
    pad_w = int(math.ceil((im_w_new - im_w) / 2))

    padded = np.pad(im, (pad_h, pad_w), 'constant', constant_values=255)
    padded_h, padded_w = padded.shape
    angle_deg = angle * 180 / math.pi
    print 'rotating to angle:', angle_deg, 'deg'
    matrix = cv2.getRotationMatrix2D((padded_w / 2, padded_h / 2), angle_deg, 1)
    result = cv2.warpAffine(padded, matrix, (padded_w, padded_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    debug_imwrite('rotated.png', result)
    return result

