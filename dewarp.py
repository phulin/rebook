import cv2
import itertools
import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import sys

from math import sqrt, cos, sin, acos, atan2, pi
from numpy.polynomial import Polynomial as P
# from scipy import interpolate
from scipy.ndimage import grey_dilation

import algorithm
import binarize
import lib

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# focal length f = 3270.5 pixels
f = 3270.5

class Line(object):
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def __call__(self, x):
        return self.m * x + self.b

    @staticmethod
    def from_polynomial(p):
        b, m = p.coeffs
        return Line(m, b)

    @staticmethod
    def from_point_slope(p0, m):
        x0, y0 = p0
        return Line(m, y0 - m * x0)

    @staticmethod
    def from_points(p0, p1):
        x0, y0 = p0
        x1, y1 = p1
        m = (y1 - y0) / (x1 - x0)
        return Line.from_point_slope(p0, m)

    def intersect(self, other):
        x = (other.b - self.b) / (self.m - other.m)
        return np.array((x, self(x)))

    def altitude(self, point):
        return Line.from_point_slope(point, -1 / self.m)

    def draw(self, im, thickness=2, color=BLUE):
        _, im_w, _ = im.shape
        cv2.line(im, (0, int(self(0))), (im_w, int(self(im_w))), color=color,
                 thickness=thickness)

    def base(self):
        return np.array((0, self.b))

    def vector(self):
        vec = np.array((1, self.m))
        vec /= np.linalg.norm(vec)
        return vec

    def polynomial(self):
        return P([self.b, self.m], domain=[])

    def offset(self, offset):
        return Line.from_point_slope(self.base() + offset, self.m)

    def closest_polynomial_intersection(self, poly, x0):
        roots = (poly - self.polynomial()).roots()
        good_roots = roots[abs(roots.imag) < 1e-10]
        x1 = good_roots[abs(good_roots - x0).argmin()]
        return np.array((x1, poly(x1)))

    @staticmethod
    def best_intersection(lines):
        bases = [l.base() for l in lines]
        vecs = [l.vec() for l in lines]
        K = len(lines)
        I = np.eye(2)
        R = K * I - sum((np.linalg.outer(v, v) for v in vecs))
        q = sum(bases) - sum((v * v.dot(a)) for v, a in zip(vecs, bases))
        R_inv = np.linalg.pinv(R)
        p = np.dot(R_inv, q)
        print p
        return p

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

def trace_baseline(im, line, model, color):
    x_min, x_max = line[0].x, line[-1].x + line[-1].w
    domain = np.arange(x_min, x_max)
    points = np.vstack([domain, model(domain)]).T
    for p in points:
        cv2.circle(im, tuple(p.astype(int)), 2, color, -1)

def remove_outliers(im, AH, lines):
    lines_debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    result = []
    models = []
    for l in lines:
        if len(l) < 5: continue

        points = np.array(map(base_point, l))
        model, inliers = skimage.measure.ransac(points, PolyModel5, 10, AH / 15.0)
        poly = model.params
        trace_baseline(lines_debug, l, poly, BLUE)
        for p, is_in in zip(points, inliers):
            if is_in:
                cv2.circle(lines_debug, tuple(p.astype(int)), 4, (0, 255, 0), -1)

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

    return list(itertools.compress(lines, full_line_mask)), vertical_lines

def centroid(poly, line):
    first, last = line[0], line[-1]
    _, x0, _, w0, _ = first
    _, x1, _, w1, _ = last
    domain = np.linspace(x0, x1 + w1, 20)
    points = np.vstack([domain, poly(domain)]).T
    return points.mean(axis=0)

def plot_norm(points, *args, **kwargs):
    norm = points - points[0]
    norm /= norm[-1][0]
    norm_T = norm.T
    norm_T[1] -= norm_T[1][0]
    # norm_T[1] /= norm_T[1].max()
    plt.plot(norm_T[0], norm_T[1], *args, **kwargs)

def base_point((c, x, y, w, h)):
    return np.array((x + w / 2.0, y + h))

def estimate_directrix(lines, models, v):
    vx, vy = v

    if vy < 0:
        # use bottom line as C0
        C0_poly, C0_line, C1_poly = models[-1], lines[-1], models[0]
    else:
        C0_poly, C0_line, C1_poly = models[0], lines[0], models[-1]

    first_letter, last_letter = C0_line[0], C0_line[-1]
    _, x0, _, w0, _ = first_letter
    _, x1, _, w1, _ = last_letter
    domain = np.linspace(x0, x1 + w1, 100)
    C_points = []
    for x0 in domain:
        y0 = C0_poly(x0)
        line = Line.from_points(v, (x0, y0))
        x1, y1 = line.closest_polynomial_intersection(C1_poly, x0)

        lam = (vy - y0) / (y1 - y0)
        alpha = 20 * lam / (20 + lam - 1)
        p0, p1 = np.array([x0, y0]), np.array([x1, y1])
        C_points.append((1 - alpha) * p0 + alpha * p1)

    C_points = np.array(C_points).T
    C = C_points.T.mean(axis=0)

    theta = acos(f / sqrt(vx ** 2 + vy ** 2 + f ** 2))
    print 'theta:', theta
    A = np.array([
        [1, C[0] / f * -sin(theta)],
        [0, cos(theta) - C[1] / f * sin(theta)]
    ])

    D_points = np.linalg.inv(A).dot(C_points)

    # plot_norm(np.vstack([domain, last_poly(domain)]).T, label='C0')
    # plot_norm(np.vstack([domain, first_poly(domain)]).T, label='C1')
    # plot_norm(C_points.T, label='C20')
    # plot_norm(D_points.T, label='D')
    # plt.axes().legend()
    # plt.show()

    return D_points

def aspect_ratio(im, lines, D, v):
    vx, vy = v

    p0_left = base_point(lines[-1][0])
    p1_left = base_point(lines[0][0])

    # Guess O.
    im_h, im_w = im.shape
    O = np.array((im_w / 2.0, im_h / 2.0))

    # TODO: stop assuming O=(0, 0). it's really (V - O) * (L - O) = 0
    m = -(vx - O[0]) / (vy - O[1])
    L0 = Line.from_point_slope(p0_left, m)
    L1 = Line.from_point_slope(p1_left, m)
    perp = L0.altitude(v)
    p0, p1 = L0.intersect(perp), L1.intersect(perp)
    h_img = np.linalg.norm(p0 - p1)

    L = Line(m, -m * O[0] - (f ** 2) / (vy - O[1]))
    F = L.altitude(v).intersect(L)
    _, x0r, y0r, w0r, h0r = lines[-1][-1]
    p0r = np.array([x0r + w0r / 2.0, y0r + h0r])
    F_C0r = Line.from_points(F, p0r)
    q0 = F_C0r.intersect(L0)
    l_img = np.linalg.norm(q0 - p0)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    L0.draw(debug)
    L1.draw(debug)
    L.draw(debug, color=GREEN)
    F_C0r.draw(debug, color=RED)
    lib.debug_imwrite('aspect.png', debug)

    # Convergence line perp to V=(vx, vy, f)
    # y = -vx / vy * x + -f^2 / vy
    alpha = atan2(np.linalg.norm(p1), f)
    theta = acos(f / sqrt(vx ** 2 + vy ** 2 + f ** 2))
    beta = pi / 2 - theta

    lp_img = abs(D[0][-1] - D[0][0])
    wp_img = np.linalg.norm(np.diff(D.T, axis=0), axis=1).sum()
    print 'h_img:', h_img, 'l\'_img:', lp_img, 'alpha:', alpha
    print 'l_img:', l_img, 'w\'_img:', wp_img, 'beta:', beta
    r = h_img * lp_img * cos(alpha) / (l_img * wp_img * cos(alpha + beta))

    return r

def dewarp(im):
    # Meng et al., Metric Rectification of Curved Document Images
    lib.debug = True
    im = binarize.binarize(im, algorithm=binarize.yan)
    im_h, im_w = im.shape

    AH = algorithm.dominant_char_height(im)
    print 'AH =', AH
    letters = algorithm.letter_contours(AH, im)
    lines = algorithm.collate_lines(AH, letters)
    lines.sort(key=lambda l: l[0].y)
    lines, verticals = side_lines(im, AH, lines)
    lines, models = remove_outliers(im, AH, lines)
    print 'full lines:', len(lines)

    p_left, p_right = verticals
    vy, = (p_left - p_right).roots()
    vx = p_left(vy)
    v0 = np.array((vx, vy))

    # TODO: make vanishing point determination optimal
    v = v0

    print 'vanishing point:', v0

    D = estimate_directrix(lines, models, v)

    r = aspect_ratio(im, lines, D, v)
    print 'aspect ratio:', r

    return r

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    dewarp(im)

if __name__ == '__main__':
    go(sys.argv)
