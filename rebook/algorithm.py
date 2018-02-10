from __future__ import division, print_function

import cv2
import math
import numpy as np
from scipy import interpolate

import lib

from lib import debug_imwrite, is_bw
from letters import Letter, TextLine

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

def skew_angle(im, orig, AH, lines):
    if len(orig.shape) == 2:
        debug = cv2.cvtColor(orig, cv2.COLOR_GRAY2RGB)
    else:
        debug = orig.copy()

    alphas = []
    for l in lines:
        if len(l) < 10: continue

        line_model = l.fit_line()
        line_model.draw(debug)
        alphas.append(line_model.angle())

    debug_imwrite('lines.png', debug)

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

def all_letters(im):
    max_label, labels, stats, centroids = \
        cv2.connectedComponentsWithStats(im ^ 255, connectivity=4)
    return [Letter(label, labels, stats[label], centroids[label]) \
            for label in range(1, max_label)]

def dominant_char_height(im, letters=None):
    if letters is None:
        letters = all_letters(im)

    heights = [letter.h for letter in letters if letter.w > 5]

    hist, _ = np.histogram(heights, 256, [0, 256])
    # TODO: make depend on DPI.
    AH = np.argmax(hist[8:]) + 8  # minimum height 8

    if lib.debug:
        debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for letter in letters:
            letter.box(debug, color=lib.GREEN if letter.h == AH else lib.RED)
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
    word_boxes = [__x_y_w_h for __x_y_w_h in word_boxes if __x_y_w_h[4] < 3 * AH and __x_y_w_h[4] > AH / 3 and __x_y_w_h[3] > AH / 3]

    return word_boxes

def valid_letter(AH, l):
    return l.h < 3 * AH and l.w < 5 * AH and l.h > AH / 2 and l.w > AH / 3

def letter_contours(AH, im, letters=None):
    if letters is None:
        letters = all_letters(im)

    if lib.debug:
        debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for l in letters:
            l.box(debug, color=lib.GREEN if valid_letter(AH, l) else lib.RED)
        lib.debug_imwrite('letters.png', debug)

    # Slightly tuned from paper (h < 3 * AH and h < AH / 4)
    return [l for l in letters if valid_letter(AH, l)]

def horizontal_lines(AH, im, components=None):
    if components is None:
        components = all_letters(im)

    result = []
    for component in components:
        if component.w > AH * 20:
            mask = component.raster()
            proj = mask.sum(axis=0)
            smooth = (proj[:-2] + proj[1:-1] + proj[2:]) / 3.0
            max_height_var = np.percentile(smooth, 98) - np.percentile(smooth, 2)
            if smooth.max() <= AH / 4.0 and max_height_var <= AH / 6.0:
                result.append(component)

    return result

def combine_underlined(AH, im, lines, components):
    lines_set = set(lines)
    underlines = horizontal_lines(AH, im, components)
    for underline in underlines:
        raster = underline.raster()
        bottom = underline.y + underline.h - 1 - raster[::-1].argmax(axis=0)
        close_lines = []
        for line in lines:
            base_points = line.base_points().astype(int)
            base_points = base_points[(base_points[:, 0] >= underline.x) \
                                      & (base_points[:, 0] < underline.right())]
            if len(base_points) == 0: continue

            base_ys = base_points[:, 1]
            underline_ys = bottom[base_points[:, 0] - underline.x]
            if np.all(np.abs(base_ys - underline_ys) < AH):
                line.underlines.append(underline)
                close_lines.append(line)

        if len(close_lines) > 1:
            # print('merging some underlined lines!')
            combined = close_lines[0]
            lines_set.remove(combined)
            for line in close_lines[1:]:
                lines_set.remove(line)
                combined.merge(line)

            lines_set.add(combined)

    return list(lines_set)

def collate_lines(AH, word_boxes):
    word_boxes = sorted(word_boxes, key=lambda c_x_y_w_h: c_x_y_w_h[1])
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
            candidates.sort(key=lambda d_l: d_l[0])
            _, line = candidates[0]
            line.append(word_box)
            # print "  selected:", x, y, w, h
        else:
            lines.append([word_box])

    return [TextLine(l) for l in lines]

def collate_lines_2(AH, word_boxes):
    word_boxes = sorted(word_boxes, key=lambda c_x_y_w_h1: c_x_y_w_h1[1])
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
    print('AH =', AH)

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

    line_coords = [x1_y1_x2_y2 for x1_y1_x2_y2 in line_coords if x1_y1_x2_y2[2] - x1_y1_x2_y2[0] > median_width * 0.8]

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
        print("warning: too much rotation")
        return im

    angle_deg = angle * 180 / math.pi
    print('rotating to angle:', angle_deg, 'deg')

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
    # im should be black-on-white. max stroke width 41.
    assert im.dtype == np.uint8 and is_bw(im)

    inv = im + 1
    inv_mask = im ^ 255
    dists = cv2.distanceTransform(inv, cv2.DIST_L2, 5)
    stroke_radius = min(20, int(math.ceil(np.percentile(dists, 95))))
    dists = 2 * dists + 1
    dists = dists.astype(np.uint8)
    rect = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    for idx in range(stroke_radius):
        dists = cv2.dilate(dists, rect)
        dists &= inv_mask

    dists_mask = (dists >= 41).astype(np.uint8) - 1
    dists &= dists_mask

    return dists

# only after rotation!
def fine_dewarp(im, lines):
    im_h, im_w = im.shape[:2]

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    points = []
    y_offsets = []
    for line in lines:
        base_points = np.array([letter.base_point() for letter in line.inliers()])
        median_y = np.median(base_points[:, 1])
        y_offsets.append(median_y - base_points[:, 1])
        points.append(base_points)
        for p in base_points:
            pt = tuple(np.round(p).astype(int))
            cv2.circle(debug, (pt[0], int(median_y)), 4, lib.RED, -1)
            cv2.circle(debug, pt, 4, lib.GREEN, -1)
    cv2.imwrite('points.png', debug)

    points = np.concatenate(points)
    y_offsets = np.concatenate(y_offsets)
    mesh = np.mgrid[:im_w, :im_h].astype(np.float32)
    xmesh, ymesh = mesh

    # y_offset_interp = interpolate.griddata(points, y_offsets, xmesh, ymesh, method='nearest')
    # y_offset_interp = y_offset_interp.clip(-5, 5)
    # mesh[1] += y_offset_interp  # (mesh[0], mesh[1], grid=False)

    y_offset_interp = interpolate.SmoothBivariateSpline(
        points[:, 0], points[:, 1], y_offsets
    )
    ymesh += y_offset_interp(xmesh, ymesh, grid=False)

    conv_xmesh, conv_ymesh = cv2.convertMaps(xmesh, ymesh, cv2.CV_16SC2)
    out = cv2.remap(im, conv_xmesh, conv_ymesh, interpolation=cv2.INTER_LINEAR).T
    cv2.imwrite('corrected.png', out)

    return out
