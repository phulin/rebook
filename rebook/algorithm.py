import cv2
import math
import numpy as np
# import numpy.polynomial.polynomial as poly

import lib
from lib import debug_imwrite, is_bw
from letters import Letter, TextLine

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def gradient(im):
    return cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)

def text_contours(im, original):
    im_h, im_w = im.shape
    min_feature_size = im_h / 150

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
            if cv2.contourArea(c) > image_area * 0.3:
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

def skew_angle(im, orig, AH, lines):
    if len(orig.shape) == 2:
        lines_debug = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    else:
        lines_debug = orig.copy()

    alphas = []
    for l in lines:
        if len(l) < 10: continue

        points = np.array([(x + w / 2, y + h) for (_, x, y, w, h) in l])
        ys = np.array([y + h for _, x, y, w, h in l])
        ys_padded = np.pad(ys, 2, 'edge')
        ys_average = (ys_padded[4:] + ys_padded[3:-1] +
                      ys_padded[1:-3] + ys_padded[:-4]) / 4
        points_masked = points[ys >= ys_average - AH / 8]
        # print 'masked:', len(points) - len(points_masked), 'out of', len(points)

        vx, vy, x1, y1 = cv2.fitLine(points_masked, cv2.DIST_L2, 0, 0.01, 0.01)
        cv2.line(lines_debug,
                    (x1 - vx * 10000, y1 - vy * 10000),
                    (x1 + vx * 10000, y1 + vy * 10000),
                    (255, 0, 0), thickness=3)
        alphas.append(math.atan2(vy, vx))

    debug_imwrite('lines.png', lines_debug)

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
    _, contours, [hierarchy] = cv2.findContours(im ^ 255, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
    letters = top_contours(contours, hierarchy)
    char_heights = [cv2.boundingRect(c)[3] for c in letters]

    hist, _ = np.histogram(char_heights, 256, [0, 256])
    # TODO: make depend on DPI.
    AH = np.argmax(hist[8:]) + 8  # minimum height 8

    if lib.debug:
        debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for c in letters:
            x, y, w, h = cv2.boundingRect(c)
            if h == AH:
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 2)
            else:
                cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 0, 255), 2)
        debug_imwrite('heights.png', debug)

    return AH

def word_contours(AH, im):
    opened = cv2.morphologyEx(im ^ 255, cv2.MORPH_OPEN, cross33)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (int(AH * 0.6) | 1, 1))
    rls = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, horiz)
    debug_imwrite('rls.png', rls)

    _, contours, [hierarchy] = cv2.findContours(rls, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
    words = top_contours(contours, hierarchy)
    word_boxes = [tuple([word] + list(cv2.boundingRect(word))) for word in words]
    # Slightly tuned from paper (h < 3 * AH and h < AH / 4)
    word_boxes = filter(
        lambda (_, x, y, w, h): h < 3 * AH and h > AH / 3 and w > AH / 3,
        word_boxes
    )

    return word_boxes

def valid_letter(AH, c, x, y, w, h):
    return h < 3 * AH and w < 3 * AH and h > AH / 2 and w > AH / 3

def letter_contours(AH, im):
    _, contours, [hierarchy] = cv2.findContours(im ^ 255, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)
    all_contours = top_contours(contours, hierarchy)
    letter_boxes = [Letter(letter, *cv2.boundingRect(letter)) for letter in all_contours]

    if lib.debug:
        debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for l in letter_boxes:
            l.box(debug, color=(0, 255, 0) if valid_letter(AH, *l) else (0, 0, 255))
        lib.debug_imwrite('letters.png', debug)

    # Slightly tuned from paper (h < 3 * AH and h < AH / 4)
    letter_boxes = filter(lambda t: valid_letter(AH, *t.tuple()), letter_boxes)

    return letter_boxes

def horizontal_lines(AH, im):
    _, contours, [hierarchy] = cv2.findContours(im ^ 255, cv2.RETR_CCOMP,
                                                cv2.CHAIN_APPROX_SIMPLE)

    result = []
    idx = 0
    while idx >= 0:
        # catch horiz lines
        c = contours[idx]
        x, y, w, h = cv2.boundingRect(c)
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [c], 0, 1,
                         thickness=cv2.FILLED, offset=(-x, -y))
        proj = mask.sum(axis=0)
        max_height_var = np.percentile(proj, 95) - np.percentile(proj, 5)
        if hierarchy[idx][2] == -1 and w > AH * 10 and max_height_var <= 3 and h < AH:
            result.append((c, x, y, w, h))

        idx = hierarchy[idx][0]

    return result

@lib.timeit
def collate_lines(AH, word_boxes):
    word_boxes = sorted(word_boxes, key=lambda (c, x, y, w, h): x)
    lines = []
    for word_box in word_boxes:
        _, x1, y1, w1, h1 = word_box
        # print "word:", x1, y1, w1, h1
        candidates = []
        for l in lines:
            _, x0, y0, w0, h0 = l[-1]
            _, x0p, y0p, w0p, h0p = l[-2] if len(l) > 1 else l[-1]
            if x1 < x0 + w0 + 4 * AH and y0 <= y1 + h1 and y1 <= y0 + h0:
                candidates.append((x1 - x0 - w0 + abs(y1 - y0), l))
            elif x1 < x0p + w0p + AH and y0p <= y1 + h1 and y1 <= y0p + h0p:
                candidates.append((x1 - x0p - w0p + abs(y1 - y0p), l))

        if candidates:
            candidates.sort(key=lambda (d, l): d)
            _, line = candidates[0]
            line.append(word_box)
            # print "  selected:", x, y, w, h
        else:
            lines.append([word_box])

    return [TextLine(l) for l in lines]

@lib.timeit
def collate_lines_2(AH, word_boxes):
    word_boxes = sorted(word_boxes, key=lambda (c, x, y, w, h): x)
    lines = []
    for word_box in word_boxes:
        _, x1, y1, w1, h1 = word_box
        # print "word:", x1, y1, w1, h1
        best_candidate = None
        best_score = 100000
        for l in lines:
            _, x0, y0, w0, h0 = l[-1]
            _, x0p, y0p, w0p, h0p = l[-2] if len(l) > 1 else l[-1]
            score = best_score
            if x1 < x0 + w0 + 4 * AH and y0 <= y1 + h1 and y1 <= y0 + h0:
                score = x1 - x0 - w0 + abs(y1 - y0)
            elif x1 < x0p + w0p + AH and y0p <= y1 + h1 and y1 <= y0p + h0p:
                score = x1 - x0p - w0p + abs(y1 - y0p)
            if score < best_score:
                best_score = score
                best_candidate = l

        if best_candidate:
            best_candidate.append(word_box)
            # print "  selected:", x, y, w, h
        else:
            lines.append([word_box])

    return [TextLine(l) for l in lines]

def dewarp_text(im):
    # Goal-Oriented Rectification (Stamatopoulos et al. 2011)
    im_h, im_w = im.shape

    AH = dominant_char_height(im)
    print 'AH =', AH

    word_boxes = word_contours(im)
    lines = collate_lines(AH, word_boxes)

    word_coords = [np.array([(x, y, x + w, y + h) for c, x, y, w, h in l]) for l in lines]
    bounds = np.array([
        word_coords[np.argmin(word_coords[:, 0]), 0],
        word_coords[np.argmin(word_coords[:, 2]), 2]
    ])
    line_coords = [(
        min((x for _, x, y, w, h in l)),
        min((y for _, x, y, w, h in l)),
        max((x + w for _, x, y, w, h in l)),
        max((y + h for _, x, y, w, h in l)),
    ) for l in lines]

    widths = np.array([x2_ - x1_ for x1_, y1_, x2_, y2_ in line_coords])
    median_width = np.median(widths)

    line_coords = filter(lambda (x1, y1, x2, y2): x2 - x1 > median_width * 0.8,
                         line_coords)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for _, x, y, w, h in word_boxes:
        cv2.rectangle(debug, (x, y), (x + w, y + h), (0, 255, 0), 1)
    for x1, y1, x2, y2 in line_coords:
        cv2.rectangle(debug, (x1, y1), (x2, y2), (255, 0, 0), 2)
    debug_imwrite('lines.png', debug)

    left = np.array([(x, y) for _, x, y, _, _ in line_coords])
    right = np.array([(x, y) for _, _, _, x, y in line_coords])
    vertical_lines = []
    bad_line_mask = np.array([False] * len(lines))
    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for coords in [left, right]:
        masked = np.ma.MaskedArray(coords, np.ma.make_mask_none(coords.shape))
        while np.ma.count(masked) > 2:
            # fit line to coords
            xs, ys = masked[:, 0], masked[:, 1]
            [c0, c1] = np.ma.polyfit(xs, ys, 1)
            diff = c0 + c1 * xs - ys
            if np.linalg.norm(diff) > AH:
                masked.mask[np.ma.argmax(masked)] = True

        vertical_lines.append((c0, c1))
        bad_line_mask |= masked.mask

        cv2.line(debug, (0, c0), (im_w, c0 + c1 * im_w), (255, 0, 0), 3)

    debug_imwrite('vertical.png', debug)

    good_lines = np.where(~bad_line_mask)
    AB = good_lines.min()
    DC = good_lines.max()

    return AB, DC, bounds

def safe_rotate(im, angle):
    debug_imwrite('prerotated.png', im)
    im_h, im_w = im.shape
    if abs(angle) > math.pi / 4:
        print "warning: too much rotation"
        return im

    angle_deg = angle * 180 / math.pi
    print 'rotating to angle:', angle_deg, 'deg'

    im_h_new = im_w * abs(math.sin(angle)) + im_h * math.cos(angle)
    im_w_new = im_h * abs(math.sin(angle)) + im_w * math.cos(angle)

    pad_h = int(math.ceil((im_h_new - im_h) / 2))
    pad_w = int(math.ceil((im_w_new - im_w) / 2))

    padded = np.pad(im, (pad_h, pad_w), 'constant', constant_values=255)
    padded_h, padded_w = padded.shape
    matrix = cv2.getRotationMatrix2D((padded_w / 2, padded_h / 2), angle_deg, 1)
    result = cv2.warpAffine(padded, matrix, (padded_w, padded_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    debug_imwrite('rotated.png', result)
    return result

def fast_stroke_width(im):
    # im should be black-on-white.
    assert im.dtype == np.uint8 and is_bw(im)

    inv = im + 1
    inv_mask = im ^ 255
    dists = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    dists = dists.astype(np.uint8)
    stroke_radius = int(math.ceil(np.percentile(dists, 95)))
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for idx in range(stroke_radius):
        dists = cv2.dilate(dists, rect)
        dists &= inv_mask

    return dists
