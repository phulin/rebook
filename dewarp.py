import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import sys

from math import sqrt, cos, sin, acos
from numpy.polynomial import Polynomial as P
# from scipy import interpolate
from scipy.ndimage import grey_dilation

import algorithm
import binarize
import lib

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

def peak_points(l, AH):
    x_min, x_max = l[0][1], l[-1][1] + l[-1][3]
    y_min = min([y for c, x, y, w, h in l]) + 1
    y_max = max([y + h for c, x, y, w, h in l]) + 1
    height, width = y_max - y_min, x_max - x_min

    mask = np.zeros((y_max - y_min, x_max - x_min))
    contours = [c for c, x, y, w, h in l]
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED,
                        offset=(-x_min, -y_min))

    old_bottom = height - mask[::-1].argmax(axis=0)
    good_bottoms = mask.max(axis=0) > 0
    bottom_xs, = np.where(good_bottoms)
    bottom_ys = old_bottom[good_bottoms]
    bottom = np.interp(np.arange(width), bottom_xs, bottom_ys)
    assert (bottom[good_bottoms] == old_bottom[good_bottoms]).all()

    delta = AH / 2
    peaks = grey_dilation(bottom, size=2 * delta + 1)
    bottom_points = np.array(zip(range(width), bottom))
    peak_points = bottom_points[bottom_points[:, 1] == peaks]
    return peak_points

def draw_piecewise_linear_model(im, AH, lines):
    lines_debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    for l in lines:
        x_min, x_max = l[0][1], l[-1][1] + l[-1][3]
        y_min = min([y for c, x, y, w, h in l]) + 1
        y_max = max([y + h for c, x, y, w, h in l]) + 1
        width = y_max - y_min, x_max - x_min

        peaks = peak_points(l, AH)
        peak_xs = peaks[:, 0]

        baseline = np.array([-1] * width)

        W = AH * 8
        for x1 in range(0, width - W, W / 2) + [width - W]:
            x2 = x1 + W
            peaks_window = peaks[np.logical_and(x1 <= peak_xs, peak_xs < x2)]
            [c0, c1] = np.polyfit(peaks_window[:, 0], peaks_window[:, 1], 1)  # ys = c1 + c0 * xs
            fitted = c1 + c0 * np.arange(x1, x2)
            b = baseline[x1:x2]
            baseline[x1:x2] = np.where(b == -1, fitted, (fitted + b) / 2)

        for x in range(x_min, x_max):
            cv2.circle(lines_debug, (x, baseline[x - x_min] + y_min), 2, (0, 0, 255), -1)

    lib.debug_imwrite('linear.png', lines_debug)

class PolyModel5(object):
    def estimate(self, data):
        self.params = P.fit(data[:, 0], data[:, 1], 5, domain=[])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 0]) - data[:, 1])

def remove_outliers(im, AH, lines):
    lines_debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    result = []
    models = []
    for l in lines:
        if len(l) < 5: continue

        points = np.array([(x + w / 2, y + h) for _, x, y, w, h, in l])
        x_min, x_max = l[0][1], l[-1][1] + l[-1][3]

        model, inliers = skimage.measure.ransac(points, PolyModel5, 10, AH / 15.0)
        poly = model.params
        for x in range(x_min, x_max + 1):
            cv2.circle(lines_debug, (x, int(poly(x))), 2, (0, 0, 255), -1)
        for p, is_in in zip(points, inliers):
            if is_in:
                cv2.circle(lines_debug, tuple(p), 4, (0, 255, 0), -1)

        result.append(list(itertools.compress(l, inliers)))
        models.append(poly)

    lib.debug_imwrite('lines.png', lines_debug)
    return result, models

# x = my + b model weighted t
class LinearXModel(object):
    def estimate(self, data):
        self.params = P.fit(data[:, 1], data[:, 0], 1, domain=[])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 1]) - data[:, 0])

def side_lines(im, AH, lines):
    im_h, im_w = im.shape

    first_letters = [l[0] for l in lines]
    last_letters = [l[-1] for l in lines]
    left_bounds = np.array([(x, y + h / 2) for _, x, y, w, h in first_letters])
    right_bounds = np.array([(x + w, y + h / 2) for _, x, y, w, h in last_letters])

    vertical_lines = []
    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for coords in [left_bounds, right_bounds]:
        model, inliers = skimage.measure.ransac(coords, LinearXModel, 4, AH / 8.0)
        vertical_lines.append(model.params)
        for p, inlier in zip(coords, inliers):
            color = GREEN if inlier else RED
            cv2.circle(debug, tuple(p), 4, color, -1)

    for p in vertical_lines:
        cv2.line(debug, (int(p(0)), 0), (int(p(im_h)), im_h), (255, 0, 0), 2)
    lib.debug_imwrite('vertical.png', debug)

    p_left, p_right = vertical_lines
    full_line_mask = np.logical_and(
        abs(p_left(left_bounds[:, 1]) - left_bounds[:, 0]) < AH,
        abs(p_right(right_bounds[:, 1]) - right_bounds[:, 0]) < AH
    )

    return vertical_lines, full_line_mask

def centroid(poly, line):
    first, last = line[0], line[-1]
    _, x0, _, w0, _ = first
    _, x1, _, w1, _ = last
    domain = np.linspace(x0, x1 + w1, 20)
    points = np.vstack([domain, poly(domain)]).T
    return points.mean(axis=0)

def dewarp(im):
    # Meng et al., Metric Rectification of Curved Document Images
    lib.debug = True
    im = binarize.binarize(im) # , algorithm=binarize.yan)

    AH = algorithm.dominant_char_height(im)
    print 'AH =', AH
    letters = algorithm.letter_contours(AH, im)
    lines = algorithm.collate_lines(AH, letters)
    lines.sort(key=lambda l: l[0][2])
    verticals, full_line_mask = side_lines(im, AH, lines)
    lines, models = remove_outliers(im, AH, lines)

    p_left, p_right = verticals
    vy, = (p_left - p_right).roots()
    vx = p_left(vy)
    v0 = np.array((vx, vy))
    print 'vanishing point:', v0
    print 'full lines:', full_line_mask.sum()

    # focal length f = 3270.5 pixels
    first_full = full_line_mask.argmax()
    # last_full = len(lines) - 1 - full_line_mask[::-1].argmax()
    first_line, first_poly = lines[first_full], models[first_full]
    first_letter, last_letter = first_line[0], first_line[-1]
    _, x0, _, w0, _ = first_letter
    _, x1, _, w1, _ = last_letter
    print first_letter[1:], last_letter[1:]
    domain = np.linspace(x0, x1 + w1, 50)
    C_points = np.vstack([domain, first_poly(domain)])
    C = C_points.T.mean(axis=0)
    plt.plot(C_points[0], -C_points[1])
    plt.show()
    f = 3270.5
    T = np.array([C[0], C[1], f])

    theta = acos(f / sqrt(vx ** 2 + vy ** 2 + f ** 2))
    A = np.array([
        [1, T[0] / T[2] * -sin(theta)],
        [0, cos(theta) - T[1] / T[2] * sin(theta)]
    ])

    D_points = np.linalg.inv(A).dot(C_points)
    plt.plot(D_points[0], D_points[1])
    plt.show()

    # TODO: make vanishing point determination optimal
    v = v0

    return v

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    dewarp(im)

if __name__ == '__main__':
    go(sys.argv)
