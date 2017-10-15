import cv2
import numpy as np
# import IPython
import math

from binarize import otsu, roth

debug = True
def debug_imwrite(path, im):
    if debug:
        cv2.imwrite(path, im)

def normalize_u8(im):
    im_max = im.max()
    alpha = 255 / im_max
    beta = im.min() * im_max / 255
    return cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

def clip_u8(im):
    return im.clip(0, 255).astype(np.uint8)

def bool_to_u8(im):
    return im.astype(np.uint8) - 1

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def gradient(im):
    return cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)

def hsl_gray(im):
    assert len(im.shape) == 3
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    _, l, s = cv2.split(hls)
    return s, l

def text_contours(im, original):
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
                                 thickness=cv2.cv.CV_FILLED,
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

    return good_contours, bad_contours

# and x > 0.02 * im_w \
# and x + w < 0.98 * im_w \
# and y > 0.02 * im_h \
# and y + h < 0.98 * im_h:

def binarize(im, algorithm=otsu, resize=1.0):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    if (im + 1 < 2).all():  # black and white
        return im
    else:
        if resize < 0.99 or resize > 1.01:
            im = cv2.resize(im, (0, 0), None, resize, resize)
        if len(im.shape) > 2:
            sat, lum = hsl_gray(im)
            # sat, lum = clahe.apply(sat), clahe.apply(lum)
            return algorithm(lum)  # & yan(l, T=35)
        else:
            # img = clahe.apply(img)
            # cv2.imwrite('clahe.png', img)
            return algorithm(im)

def skew_angle(im):
    im_h, _ = im.shape

    first_pass = binarize(im, algorithm=roth, resize=1000.0 / im_h)

    grad = gradient(first_pass)
    space_width = (im_h / 50) | 1
    line_height = (im_h / 400) | 1
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (space_width, line_height))
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, horiz)

    lines = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)
    line_contours, _ = text_contours(grad, first_pass)
    alphas = []
    for c in line_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 * h:
            vx, vy, x1, y1 = cv2.fitLine(c, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
            cv2.line(lines,
                     (x1 - vx * 1000, y1 - vy * 1000),
                     (x1 + vx * 1000, y1 + vy * 1000),
                     (255, 0, 0), thickness=3)
            alphas.append(math.atan2(vy, vx))
    debug_imwrite('lines.png', lines)
    return np.median(alphas)

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

