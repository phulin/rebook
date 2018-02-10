from __future__ import print_function

import cv2
import itertools
import numpy as np
import scipy
import sys

from math import sqrt, cos, sin, acos, atan2, pi
from numpy import newaxis
from numpy.linalg import norm, inv
from numpy.polynomial import Polynomial as Poly
from scipy import interpolate
from scipy import optimize as opt
from scipy.ndimage import grey_dilation
from skimage.measure import ransac

import algorithm
import binarize
import collate
from geometry import Crop, Line, Line3D
import lib
from lib import RED, GREEN, BLUE
import newton

# focal length f = 3270.5 pixels
f = 3270.5
Of = np.array([0, 0, f], dtype=np.float64)

def compress(l, flags):
    return list(itertools.compress(l, flags))

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
    bottom_points = np.array(list(zip(list(range(width)), bottom)))
    peak_points = bottom_points[bottom_points[:, 1] == peaks]
    return peak_points

class PolyModel5(object):
    def estimate(self, data):
        self.params = Poly.fit(data[:, 0], data[:, 1], 5, domain=[-1, 1])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 0]) - data[:, 1])

def trace_baseline(im, line, color=BLUE):
    domain = np.arange(line.left() - 100, line.right() + 100)
    points = np.vstack([domain, line.model(domain)]).T
    for p in points:
        cv2.circle(im, tuple(p.astype(int)), 2, color, -1)

def merge_lines(AH, lines):
    out_lines = [lines[0]]

    for line in lines[1:]:
        x_min, x_max = line[0].left(), line[-1].right()
        integ = (out_lines[-1].model - line.model).integ()
        if abs(integ(x_max) - integ(x_min)) / (x_max - x_min) < AH / 8.0:
            out_lines[-1].merge(line)
            points = np.array([letter.base_point() for letter in out_lines[-1]])
            new_model, inliers = ransac(points, PolyModel5, 10, AH / 15.0)
            out_lines[-1].compress(inliers)
            out_lines[-1].model = new_model.params
        else:
            out_lines.append(line)

    # debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    # for l in out_lines:
    #     trace_baseline(debug, l, BLUE)
    # lib.debug_imwrite('merged.png', debug)

    print('original lines:', len(lines), 'merged lines:', len(out_lines))
    return out_lines

@lib.timeit
def remove_outliers(im, AH, lines):
    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    result = []
    for l in lines:
        if len(l) < 5: continue

        points = np.array([letter.base_point() for letter in l])
        model, inliers = ransac(points, PolyModel5, 10, AH / 10.0)
        poly = model.params
        l.model = poly
        # trace_baseline(debug, l, BLUE)
        for p, is_in in zip(points, inliers):
            color = GREEN if is_in else RED
            cv2.circle(debug, tuple(p.astype(int)), 4, color, -1)

        l.compress(inliers)
        result.append(l)

    for l in result:
        cv2.circle(debug, tuple(l[0].left_mid().astype(int)), 6, BLUE, -1)
        cv2.circle(debug, tuple(l[-1].right_mid().astype(int)), 6, BLUE, -1)

    lib.debug_imwrite('lines.png', debug)
    return merge_lines(AH, result)

# x = my + b model weighted t
class LinearXModel(object):
    def estimate(self, data):
        self.params = Poly.fit(data[:, 1], data[:, 0], 1, domain=[-1, 1])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 1]) - data[:, 0])

def side_lines(AH, lines):
    im_h, _ = bw.shape

    left_bounds = np.array([l[0].left_mid() for l in lines])
    right_bounds = np.array([l[-1].right_mid() for l in lines])

    vertical_lines = []
    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for coords in [left_bounds, right_bounds]:
        model, inliers = ransac(coords, LinearXModel, 3, AH / 10.0)
        vertical_lines.append(model.params)
        for p, inlier in zip(coords, inliers):
            color = GREEN if inlier else RED
            cv2.circle(debug, tuple(p.astype(int)), 4, color, -1)

    for p in vertical_lines:
        cv2.line(debug, (int(p(0)), 0), (int(p(im_h)), im_h), (255, 0, 0), 2)
    lib.debug_imwrite('vertical.png', debug)

    return vertical_lines

def estimate_vanishing(AH, lines):
    p_left, p_right = side_lines(AH, lines)
    vy, = (p_left - p_right).roots()
    return np.array((p_left(vy), vy))

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

def C0_C1(lines, v):
    _, vy = v
    # use bottom line as C0 if vanishing point above image
    C0, C1 = (lines[-1], lines[0]) if vy < 0 else (lines[0], lines[-1])
    return C0, C1

def widest_domain(lines, v, n_points):
    C0, C1 = C0_C1(lines, v)

    v_lefts = [Line.from_points(v, l[0].left_bot()) for l in lines if l is not C0]
    v_rights = [Line.from_points(v, l[-1].right_bot()) for l in lines if l is not C0]
    C0_lefts = [l.text_line_intersect(C0)[0] for l in v_lefts]
    C0_rights = [l.text_line_intersect(C0)[0] for l in v_rights]

    x_min = min(C0.left(), min(C0_lefts))
    x_max = max(C0.left(), max(C0_rights))
    domain = np.linspace(x_min, x_max, n_points)

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for l in lines:
        cv2.line(debug, tuple(l[0].left_bot().astype(int)),
                tuple(l[-1].right_bot().astype(int)), GREEN, 2)
    Line.from_points(v, (x_min, C0(x_min))).draw(debug)
    Line.from_points(v, (x_max, C0(x_max))).draw(debug)
    lib.debug_imwrite('domain.png', debug)

    return domain, C0, C1

def arc_length_points(xs, ys, n_points):
    arc_points = np.stack((xs, ys))
    arc_lengths = norm(np.diff(arc_points, axis=1), axis=0)
    cumulative_arc = np.hstack([[0], np.cumsum(arc_lengths)])
    D = interpolate.interp1d(cumulative_arc, arc_points, assume_sorted=True)

    total_arc = cumulative_arc[-1]
    print('total D arc length:', total_arc)
    s_domain = np.linspace(0, total_arc, n_points)
    return D(s_domain), total_arc

N_POINTS = 200
MU = 30
def estimate_directrix(lines, v, n_points_w):
    vx, vy = v

    domain, C0, C1 = widest_domain(lines, v, N_POINTS)

    C0_points = np.vstack([domain, C0(domain)])
    longitudes = [Line.from_points(v, p) for p in C0_points.T]
    C1_points = np.array([l.closest_poly_intersect(C1.model, p) \
                          for l, p in zip(longitudes, C0_points.T)]).T
    lambdas = (vy - C0_points[1]) / (C1_points[1] - C0_points[1])
    alphas = MU * lambdas / (MU + lambdas - 1)
    C_points = (1 - alphas) * C0_points + alphas * C1_points
    C = C_points.T.mean(axis=0)

    theta = acos(f / sqrt(vx ** 2 + vy ** 2 + f ** 2))
    print('theta:', theta)
    A = np.array([
        [1, C[0] / f * -sin(theta)],
        [0, cos(theta) - C[1] / f * sin(theta)]
    ])

    D_points = inv(A).dot(C_points)
    D_points_arc, _ = arc_length_points(D_points)
    C_points_arc = A.dot(D_points_arc)

    # plot_norm(np.vstack([domain, C0(domain)]).T, label='C0')
    # plot_norm(np.vstack([domain, C1(domain)]).T, label='C1')
    # plot_norm(C_points.T, label='C20')
    # plot_norm(D_points.T, label='D')
    # plot_norm(C_points_arc.T, label='C')
    # # plt.plot(C_points.T, label='C20')
    # # plt.plot(C_points_arc.T, label='C')
    # plt.axes().legend()
    # plt.show()

    return D_points_arc, C_points_arc

def aspect_ratio(im, lines, D, v, O):
    vx, vy = v
    C0, C1 = C0_C1(lines, v)

    im_h, im_w = im.shape

    m = -(vx - O[0]) / (vy - O[1])
    L0 = Line.from_point_slope(C0.first_base(), m)
    L1 = Line.from_point_slope(C1.first_base(), m)
    perp = L0.altitude(v)
    p0, p1 = L0.intersect(perp), L1.intersect(perp)
    h_img = norm(p0 - p1)

    L = Line(m, -m * O[0] - (f ** 2) / (vy - O[1]))
    F = L.altitude(v).intersect(L)
    _, x0r, y0r, w0r, h0r = lines[-1][-1]
    p0r = np.array([x0r + w0r / 2.0, y0r + h0r])
    F_C0r = Line.from_points(F, p0r)
    q0 = F_C0r.intersect(L0)
    l_img = norm(q0 - p0)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    L0.draw(debug)
    L1.draw(debug)
    L.draw(debug, color=GREEN)
    F_C0r.draw(debug, color=RED)
    lib.debug_imwrite('aspect.png', debug)

    # Convergence line perp to V=(vx, vy, f)
    # y = -vx / vy * x + -f^2 / vy
    alpha = atan2(norm(p1 - O), f)
    theta = acos(f / sqrt((vx - O[0]) ** 2 + (vy - O[1]) ** 2 + f ** 2))
    beta = pi / 2 - theta

    lp_img = abs(D[0][-1] - D[0][0])
    wp_img = norm(np.diff(D.T, axis=0), axis=1).sum()
    print('h_img:', h_img, 'l\'_img:', lp_img, 'alpha:', alpha)
    print('l_img:', l_img, 'w\'_img:', wp_img, 'beta:', beta)
    r = h_img * lp_img * cos(alpha) / (l_img * wp_img * cos(alpha + beta))

    return r

class MuMode(object):
    def __init__(self, val):
        self.val = val

    def __eq__(self, other):
        return self.val == other.val

    def index(self):
        return 0 if self.val else -1

    def point(self, l):
        if self.val:
            return l.top_point()  # + np.array([0, -20])
        else:
            return l.base_point()  # + np.array([0, 20])

MuMode.BOTTOM = MuMode(False)
MuMode.TOP = MuMode(True)

# find mu necessary to entirely cover line with mesh
def necessary_mu(C0, C1, v, all_lines, mode):
    vx, vy = v

    line = all_lines[mode.index()]
    points = np.array([mode.point(l) for l in line])
    for p in points:
        global mu_debug
        cv2.circle(mu_debug, tuple(p.astype(int)), 6, GREEN, -1)

    longitudes = [Line.from_points(v, p) for p in points]
    C0_points = np.array([l.text_line_intersect(C0) for l in longitudes]).T
    C1_points = np.array([l.text_line_intersect(C1) for l in longitudes]).T
    lambdas = (vy - C0_points[1]) / (C1_points[1] - C0_points[1])
    alphas = (points[:, 1] - C0_points[1]) / (C1_points[1] - C0_points[1])
    mus = alphas * (1 - lambdas) / (alphas - lambdas)

    return mus.max() + 0.01 if np.median(mus) >= 0.5 else mus.min() - 0.01

@lib.timeit
def generate_mesh(all_lines, lines, C_arc, v, n_points_h):
    vx, vy = v
    C_arc_T = C_arc.T

    C0, C1 = C0_C1(lines, v)

    # first, calculate necessary mu.
    global mu_debug
    mu_debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    mu_bottom = necessary_mu(C0, C1, v, all_lines, MuMode.BOTTOM)
    mu_top = necessary_mu(C0, C1, v, all_lines, MuMode.TOP)
    lib.debug_imwrite('mu.png', mu_debug)

    longitude_lines = [Line.from_points(v, p) for p in C_arc_T]
    longitudes = []
    mus = np.linspace(mu_top, mu_bottom, n_points_h)
    for l, C_i in zip(longitude_lines, C_arc_T):
        p0 = l.closest_poly_intersect(C0.model, C_i)
        p1 = l.closest_poly_intersect(C1.model, C_i)
        lam = (vy - p0[1]) / (p1[1] - p0[1])
        alphas = mus * lam / (mus + lam - 1)
        longitudes.append(np.outer(1 - alphas, p0) + np.outer(alphas, p1))

    result = np.array(longitudes)

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for l in result[::50]:
        for p in l[::50]:
            cv2.circle(debug, tuple(p.astype(int)), 6, BLUE, -1)
    trace_baseline(debug, C0, RED)
    trace_baseline(debug, C1, RED)
    lib.debug_imwrite('mesh.png', debug)

    return np.array(longitudes).transpose(1, 0, 2)

@lib.timeit
def correct_geometry(orig, mesh, interpolation=cv2.INTER_LINEAR):
    # coordinates (u, v) on mesh -> mesh[u][v] = (x, y) in distorted image
    mesh32 = mesh.astype(np.float32)
    xmesh, ymesh = mesh32[:, :, 0], mesh32[:, :, 1]
    conv_xmesh, conv_ymesh = cv2.convertMaps(xmesh, ymesh, cv2.CV_16SC2)
    out = cv2.remap(orig, conv_xmesh, conv_ymesh, interpolation=interpolation,
                    borderMode=cv2.BORDER_REPLICATE)
    lib.debug_imwrite('corrected.png', out)

    return out

def spline_model(line):
    base_points = np.array([letter.base_point() for letter in line])
    _, indices = np.unique(base_points[:, 0], return_index=True)
    data = base_points[indices]
    return interpolate.UnivariateSpline(data[:, 0], data[:, 1])

def valid_curvature(line):
    if len(line) < 4: return True

    poly = spline_model(line)
    polyp = poly.derivative()
    polypp = polyp.derivative()

    x_range = line.left(), line.right()
    x_points = np.linspace(x_range[0], x_range[1], 50)

    curvature = abs(polypp(x_points)) / (1 + polyp(x_points) ** 2)  # ** 3/2
    # print 'curvature:', curvature.max()

    global curvature_debug
    for p in zip(x_points, poly(x_points)):
        cv2.circle(curvature_debug, (int(p[0]), int(p[1])), 2, BLUE, -1)
    return curvature.max() < 0.3

def min_crop(lines):
    box = Crop(
        min([line.left() for line in lines]),
        min([letter.y for letter in lines[0]]),
        max([line.right() for line in lines]),
        max([letter.y + letter.h for letter in lines[-1]]),
    )
    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    box.draw(debug)
    lib.debug_imwrite('crop.png', debug)
    return box

@lib.timeit
def dewarp_fine(im):
    lib.debug_prefix = 'fine_'

    AH, all_lines, lines = get_AH_lines(im)

    points = []
    offsets = []
    for line in lines:
        bases = np.array([l.base_point() for l in line])
        median_y = np.median(bases[:, 1])
        points.extend(bases)
        offsets.extend(median_y - bases[:, 1])

    points = np.array(points)
    offsets = np.array(offsets)

    im_h, im_w = im.shape
    # grid_x, grid_y = np.mgrid[:im_h, :im_w]
    # y_offset_interp = interpolate.griddata(points, offsets,
    #                                        (grid_x, grid_y), method='nearest')
    y_offset_interp = interpolate.SmoothBivariateSpline(
        points[:, 0], points[:, 1], offsets
    )

    new = np.full(im.shape, 0, dtype=np.uint8)
    _, contours, [hierarchy] = \
        cv2.findContours(im, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    draw_contours(new, contours, hierarchy, y_offset_interp, 0, 255)

    lib.debug_imwrite('fine.png', new)

    return new

def draw_contours(im, contours, hierarchy, y_offset_interp, idx, color,
                  depth=0, passed_offset=None):
    while idx >= 0:
        x, y, w, h = cv2.boundingRect(contours[idx])
        # print '+' * depth, idx, 'color:', color, '@', x, y, 'offset:', offset, 'area:', w * h
        if passed_offset is None:
            offset = (0, -int(round(y_offset_interp(x + w / 2.0, y + h))))
            # offset = (0, -int(round(y_offset_interp[y + h - 1, x + w / 2 - 1])))
        else:
            offset = passed_offset

        cv2.drawContours(im, contours, idx, color, thickness=cv2.FILLED,
                         offset=offset)
        child = hierarchy[idx][2]
        if child >= 0:
            pass_offset = offset if color == 0 and w * h < 5000 else None
            draw_contours(im, contours, hierarchy, y_offset_interp, child,
                          255 - color, depth=depth + 1, passed_offset=pass_offset)
        idx = hierarchy[idx][0]

def full_lines(AH, lines, v):
    C0 = max(lines, key=lambda l: l.right() - l.left())

    v_lefts = [Line.from_points(v, l[0].left_bot()) for l in lines if l is not C0]
    v_rights = [Line.from_points(v, l[-1].right_bot()) for l in lines if l is not C0]
    C0_lefts = [l.text_line_intersect(C0)[0] for l in v_lefts]
    C0_rights = [l.text_line_intersect(C0)[0] for l in v_rights]

    mask = np.logical_and(C0_lefts <= C0.left() + AH, C0_rights >= C0.right() - AH)
    return compress(lines, mask)

def get_AH_lines(im):
    all_letters = algorithm.all_letters(im)
    AH = algorithm.dominant_char_height(im, letters=all_letters)
    print('AH =', AH)
    letters = algorithm.letter_contours(AH, im, letters=all_letters)
    print('collating...')
    all_lines = lib.timeit(collate.collate_lines)(AH, letters)
    all_lines.sort(key=lambda l: l[0].y)

    print('combining...')
    combined = algorithm.combine_underlined(AH, im, all_lines, all_letters)

    lines = remove_outliers(im, AH, combined)

    # if lib.debug:
    #     debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    #     for l in all_lines:
    #         for l1, l2 in zip(l, l[1:]):
    #             cv2.line(debug, tuple(l1.base_point().astype(int)),
    #                     tuple(l2.base_point().astype(int)), RED, 2)
    #     lib.debug_imwrite('all_lines.png', debug)

    return AH, combined, lines

N_LONGS = 15
def vanishing_point(lines, v0, O):
    C0 = lines[-1] if v0[1] < 0 else lines[0]
    others = lines[:-1] if v0[1] < 0 else lines[1:]

    domain = np.linspace(C0.left(), C0.right(), N_LONGS + 2)[1:-1]
    C0_points = np.array([domain, C0.model(domain)]).T
    longitudes = [Line.from_points(v0, p) for p in C0_points]

    lefts = [longitudes[0].text_line_intersect(line)[0] for line in others]
    rights = [longitudes[-1].text_line_intersect(line)[0] for line in others]
    valid_mask = [line.left() <= L and R < line.right() \
                   for line, L, R in zip(others, lefts, rights)]

    valid_lines = [C0] + compress(others, valid_mask)
    derivs = [line.model.deriv() for line in valid_lines]
    print('valid lines:', len(others))

    convergences = []
    for longitude in longitudes:
        intersects = [longitude.text_line_intersect(line) for line in valid_lines]
        tangents = [Line.from_point_slope(p, d(p[0])) \
                    for p, d in zip(intersects, derivs)]
        convergences.append(Line.best_intersection(tangents))

    # x vx + y vy + f^2 = 0
    # m = -vx / vy
    # b = -f^2 / vy


    L = Line.fit(convergences)
    # shift into O-origin coords
    L_O = L.offset(-O)
    vy = -(f ** 2) / L_O.b
    vx = -vy * L_O.m
    v = np.array((vx, vy)) + O

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for t in tangents: t.draw(debug, color=RED)
    for longitude in longitudes:
        longitude.draw(debug)
    L.draw(debug, color=GREEN)
    lib.debug_imwrite('vanish.png', debug)

    return v, f, L

def dewarp(orig):
    # Meng et al., Metric Rectification of Curved Document Images
    lib.debug = True
    im = binarize.binarize(orig, algorithm=binarize.ntirogiannis2014)
    global bw
    bw = im
    im_h, im_w = im.shape

    AH, all_lines, lines = get_AH_lines(im)

    v0 = estimate_vanishing(AH, lines)

    O = np.array((im_w / 2.0, im_h / 2.0))
    v = v0
    print('vanishing point:', v)
    for i in range(5):
        v, L = vanishing_point(lines, v, O)
        print('vanishing point:', v)

    lines = full_lines(AH, lines, v)

    box = min_crop(all_lines)
    D, C_arc = estimate_directrix(lines, v, box.w)

    r = aspect_ratio(im, lines, D, v, O)
    print('aspect ratio H/W:', r)
    print('fixing to 1.7')
    r = 1.7  # TODO: fix

    print('generating mesh...')
    mesh = generate_mesh(all_lines, lines, C_arc, v, r * box.w)

    print('dewarping...')
    dewarped = correct_geometry(orig, mesh)

    # print 'binarizing...'
    # dewarped_bw = binarize.binarize(dewarped, algorithm=lambda im: binarize.yan(im, alpha=0.3))

    # print 'fine dewarping...'
    # fine = dewarp_fine(dewarped_bw)

    return dewarped

# rotation matrix for rotation by ||theta|| around axis theta
# theta: 3component x N; return: 3 x 3matrix x N
def R_theta(theta):
    # these are all N-vectors
    T = norm(theta, axis=0)
    t1, t2, t3 = theta / T
    c, s = np.cos(T / 2), np.sin(T / 2)
    ss = s * s
    cs = c * s

    return np.array([
        [2 * (t1 * t1 - 1) * ss + 1,
         2 * t1 * t2 * ss - 2 * t3 * cs,
         2 * t1 * t3 * ss + 2 * t2 * cs],
        [2 * t1 * t2 * ss + 2 * t3 * cs,
         2 * (t2 * t2 - 1) * ss + 1,
         2 * t2 * t3 * ss - 2 * t1 * cs],
        [2 * t1 * t2 * ss - 2 * t2 * cs,
         2 * t2 * t3 * ss + 2 * t1 * cs,
         2 * (t3 * t3 - 1) * ss + 1]
    ])

FOCAL_PLANE_Z = -f
T0 = FOCAL_PLANE_Z / f
def image_to_focal_plane(points, O):
    if type(points) != np.ndarray:
        points = np.array(points)

    assert points.shape[0] == 2
    return np.concatenate((
        points - O[:, newaxis],
        np.full(points.shape[1:], FOCAL_PLANE_Z)[newaxis, ...]
    )).astype(np.float64)

# points: 3 x ... array of points
def project_to_image(points, O):
    assert points.shape[0] == 3
    projected = (points * FOCAL_PLANE_Z / points[2])[0:2]
    return (projected.T + O).T

# points: 3 x ... array of points
def gcs_to_image(points, O, R):
    # invert R(pt - Of)
    assert points.shape[0] == 3
    image_coords = np.tensordot(inv(R), points, axes=1)
    image_coords_T = image_coords.T
    image_coords_T += Of
    return project_to_image(image_coords, O)

# O: two-dimensional origin (middle of image/principal point)
# returns points on focal plane
def base_points(line, O):
    return image_to_focal_plane(np.array([l.base_point() for l in line]).T, O)

# l_m = fake parameter representing line position
# line_points = text base points on focal plane
def E_str(theta, g, l_m, line_points):
    # print '    theta:', theta
    # print '    a_m:', g.coef
    R = R_theta(theta)

    residuals = []
    for points, l_k in zip(line_points, l_m):
        _, (_, Ys, _) = newton.t_i_k(R, g, points, T0)
        residuals.append(Ys - l_k)

    result = np.concatenate(residuals)
    return result

DEGREE = 18
def unpack_args(args):
    # theta: 3; a_m: DEGREE; align: 2; l_m: len(lines)
    return np.split(np.array(args), (3, 3 + DEGREE, 3 + DEGREE + 2))

def E_str_packed(args, mid_points, line_points):
    theta, a_m, _, l_m = unpack_args(args)
    g = Poly(np.hstack([[0], a_m]))
    return E_str(theta, g, l_m, line_points)

def E_0(*args):
    result = E_str_packed(*args)
    print('norm:', norm(result))
    return result

def dR_dthetai(theta, R, i):
    T = norm(theta)
    inc = T / 8192
    delta = np.zeros(3)
    delta[i] = inc
    Rp = R_theta(theta + delta)
    Rm = R_theta(theta - delta)
    return (Rp - Rm) / (2 * inc)

def dR_dtheta(theta, R):
    return np.array([dR_dthetai(theta, R, i) for i in range(3)])

def dti_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface):
    R1, _, R3 = R
    dR1, dR3 = dR[:, 0], dR[:, 2]
    dR13, dR33 = dR[:, 0, 2], dR[:, 2, 2]

    Xs, _, _ = all_surface

    # dR: 3derivs x r__; dR[:, 0]: 3derivs x r1_; points: 3comps x Npoints
    # A: 3 x Npoints
    A1 = dR1.dot(all_points) * all_ts
    A2 = -dR13 * f
    A = A1 + A2[:, newaxis]
    B = R1.dot(all_points)
    C1 = dR3.dot(all_points) * all_ts  # 3derivs x Npoints
    C2 = -dR33 * f
    C = C1 + C2[:, newaxis]
    D = R3.dot(all_points)
    slopes = gp(Xs)
    return -(C - slopes * A) / (D - slopes * B)

def dE_str_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface):
    _, R2, _ = R
    dR2 = dR[:, 1]
    dR23 = dR[:, 1, 2]

    dt = dti_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface)

    term1 = dR2.dot(all_points) * all_ts
    term2 = R2.dot(all_points) * dt
    term3 = -dR23 * f

    return term1.T + term2.T + term3

def dti_dam(theta, R, g, gp, all_points, all_ts, all_surface):
    R1, R2, R3 = R

    Xs, _, _ = all_surface

    powers = np.vstack([Xs ** m for m in range(1, DEGREE + 1)])
    denom = R3.dot(all_points) - gp(Xs) * R1.dot(all_points)

    return powers / denom

def dE_str_dam(theta, R, g, gp, all_points, all_ts, all_surface):
    R1, R2, R3 = R

    dt = dti_dam(theta, R, g, gp, all_points, all_ts, all_surface)

    return (R2.dot(all_points) * dt).T

def dE_str_dl_k(line_points):
    blocks = [np.full((l.shape[-1], 1), -1) for l in line_points]
    return scipy.linalg.block_diag(*blocks)

def debug_plot_g(g, line_ts_surface):
    import matplotlib.pyplot as plt
    all_points_XYZ = np.concatenate([points for _, points in line_ts_surface],
                                    axis=1)
    domain = np.linspace(all_points_XYZ[0].min(), all_points_XYZ[0].max(), 100)
    plt.plot(domain, g(domain))
    # domain = np.linspace(-im_w / 2, im_w / 2, 100)
    # plt.plot(domain, g(domain))
    plt.show()

def Jac_E_str(args, mid_points, line_points):
    theta, a_m, _, l_m = unpack_args(args)
    R = R_theta(theta)
    dR = dR_dtheta(theta, R)
    g = Poly(np.hstack([[0], a_m]))
    gp = g.deriv()
    line_ts_surface = [newton.t_i_k(R, g, points, T0) for points in line_points]

    all_points = np.concatenate(line_points, axis=1)
    all_ts = np.concatenate([ts for ts, _ in line_ts_surface])
    all_surface = np.concatenate([surface for _, surface in line_ts_surface],
                                 axis=1)

    return np.concatenate((
        dE_str_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface),
        dE_str_dam(theta, R, g, gp, all_points, all_ts, all_surface),
        np.zeros((all_ts.shape[0], 2)),
        dE_str_dl_k(line_points),
    ), axis=1)

def debug_jac(theta, R, g, l_m, line_points, line_ts_surface):
    dR = dR_dtheta(theta, R)
    gp = g.deriv()

    all_points = np.concatenate(line_points, axis=1)
    all_ts = np.concatenate([ts for ts, _ in line_ts_surface])
    all_surface = np.concatenate([surface for _, surface in line_ts_surface], axis=1)

    print(dE_str_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface).T)
    for i in range(3):
        delta = np.zeros(3)
        inc = norm(theta) / 4096
        delta[i] = inc
        diff = E_str(theta + delta, g, l_m, line_points) - E_str(theta - delta, g, l_m, line_points)
        print(diff / (2 * inc))

    print(dE_str_dam(theta, R, g, gp, all_points, all_ts, all_surface).T)
    for i in range(1, DEGREE + 1):
        delta = np.zeros(DEGREE + 1)
        inc = g.coef[i] / 4096
        delta[i] = inc
        diff = E_str(theta, Poly(g.coef + delta), l_m, line_points) \
            - E_str(theta, Poly(g.coef - delta), l_m, line_points)
        print(diff / (2 * inc))

def E_align(theta, g, align, mid_points):
    R = R_theta(theta)

    all_points = mid_points.reshape(3, -1)
    _, (Xs, _, _) = newton.t_i_k(R, g, all_points, T0)
    Xs.shape = (2, -1)  # 2 x N

    # print (Xs - align[:, newaxis]).flatten().astype(int)
    return (Xs - align[:, newaxis]).flatten()

def E_align_packed(args, mid_points, line_points):
    theta, a_m, align, l_m = unpack_args(args)
    g = Poly(np.hstack([[0], a_m]))
    return E_align(theta, g, align, mid_points)

def dE_align_dam(theta, R, g, gp, all_points, all_ts, all_surface):
    R1, _, _ = R

    dt = dti_dam(theta, R, g, gp, all_points, all_ts, all_surface)

    return (R1.dot(all_points) * dt).T

def dE_align_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface):
    R1, _, _ = R
    dR1 = dR[:, 0]
    dR13 = dR[:, 0, 2]

    dt = dti_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface)

    term1 = dR1.dot(all_points) * all_ts
    term2 = R1.dot(all_points) * dt
    term3 = -dR13 * f

    return term1.T + term2.T + term3

def Jac_E_align(args, mid_points, line_points):
    theta, a_m, _, _ = unpack_args(args)
    R = R_theta(theta)
    dR = dR_dtheta(theta, R)
    g = Poly(np.hstack([[0], a_m]))
    gp = g.deriv()

    N = mid_points.shape[-1]  # number of lines; 2N residuals

    all_points = mid_points.reshape(3, -1)
    assert all_points.shape == (3, 2 * N)
    all_ts, all_surface = newton.t_i_k(R, g, all_points, T0)

    return np.concatenate((
        dE_align_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface),
        dE_align_dam(theta, R, g, gp, all_points, all_ts, all_surface),
        np.tile([[-1, 0], [0, -1]], (N, 1)),
        np.zeros((2 * N, len(line_points)))
    ), axis=1)

LAMBDA_2 = 0.3
def E_2(*args):
    E_str_out = E_str_packed(*args)
    E_align_out = LAMBDA_2 * E_align_packed(*args)
    result = np.concatenate([E_str_out, E_align_out])
    print('norm:', norm(result), '=', norm(E_str_out), '+', norm(E_align_out))
    return result

def Jac_E_2(*args):
    return np.concatenate([Jac_E_str(*args),
                           LAMBDA_2 * Jac_E_align(*args)])

def make_mesh_XYZ(xs, ys, g):
    return np.array([
        np.tile(xs, [len(ys), 1]),
        np.tile(ys, [len(xs), 1]).T,
        np.tile(g(xs), [len(ys), 1])
    ])

def normalize_theta(theta):
    angle = norm(theta)
    quot = int(angle / (2 * pi))
    mod = angle - 2 * pi * quot
    return theta * (mod / angle)

def debug_print_points(filename, points, step=None, color=BLUE):
    if lib.debug:
        debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        if step is not None:
            points = points[[np.s_[:]] + [np.s_[::step]] * (points.ndim - 1)]
        for p in points.reshape(2, -1).T:
            cv2.circle(debug, tuple(p.astype(int)), 2, color, -1)
        lib.debug_imwrite(filename, debug)

@lib.timeit
def make_mesh_2d(all_lines, O, R, g):
    all_letters = np.concatenate([line.letters for line in all_lines])
    corners_2d = np.concatenate([letter.corners() for letter in all_letters]).T
    corners = image_to_focal_plane(corners_2d, O)
    _, corners_XYZ = newton.t_i_k(R, g, corners, T0)

    corners_X, _, corners_Z = corners_XYZ
    relative_Z_error = (g(corners_X) - corners_Z) / corners_Z
    corners_XYZ = corners_XYZ[:, relative_Z_error <= 0.02]

    debug_print_points('corners.png', corners_2d)

    box_XYZ = Crop.from_points(corners_XYZ[:2]).expand(0.01)
    print('box_XYZ:', box_XYZ)

    # 70th percentile line width a good guess
    n_points_w = 1.2 * np.percentile(np.array([line.width() for line in all_lines]), 90)
    mesh_XYZ_x = np.linspace(box_XYZ.x0, box_XYZ.x1, 400)
    mesh_XYZ_z = g(mesh_XYZ_x)
    mesh_XYZ_xz_arc, total_arc = arc_length_points(mesh_XYZ_x, mesh_XYZ_z,
                                                   n_points_w)
    mesh_XYZ_x_arc, _ = mesh_XYZ_xz_arc

    # TODO: think more about estimation of aspect ratio for mesh
    n_points_h = n_points_w * box_XYZ.h / total_arc
    # n_points_h = n_points_w * 1.7

    mesh_XYZ_y = np.linspace(box_XYZ.y0, box_XYZ.y1, n_points_h)
    mesh_XYZ = make_mesh_XYZ(mesh_XYZ_x_arc, mesh_XYZ_y, g)
    mesh_2d = gcs_to_image(mesh_XYZ, O, R)
    print('mesh:', Crop.from_points(mesh_2d))

    debug_print_points('mesh1.png', mesh_2d, step=20)

    # make sure meshes are not reversed
    if mesh_2d[0, :, 0].mean() > mesh_2d[0, :, -1].mean():
        mesh_2d = mesh_2d[:, :, ::-1]

    if mesh_2d[1, 0].mean() > mesh_2d[1, -1].mean():
        mesh_2d = mesh_2d[:, ::-1, :]

    return mesh_2d.transpose(1, 2, 0)

def initial_args(lines, O, theta_0=(1e-7, 1e-7, 1e-7)):
    # flat surface as initial guess.
    # NB: coeff 0 forced to 0 here. not included in opt.
    a_m_0 = [0] * DEGREE
    g_0 = Poly([0] + a_m_0)

    R_0 = R_theta(theta_0)
    _, ROf_y, ROf_z = R_0.dot(Of)

    # line points on focal plane
    line_points = [base_points(line, O) for line in lines]

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for line in line_points:
        for p in line.T:
            cv2.circle(debug, tuple(project_to_image(p, O).astype(int)), 2, lib.GREEN, -1)

    # make underlines straight as well
    underlines = sum([line.underlines for line in lines], [])
    print('underlines:', len(underlines))
    for underline in underlines:
        mid_contour = (underline.top_contour() + underline.bottom_contour()) / 2
        all_mid_points = np.stack([
            underline.x + np.arange(underline.w), mid_contour,
        ])
        mid_points = all_mid_points[:, ::5]
        for p in mid_points.T:
            cv2.circle(debug, tuple(p.astype(int)), 4, lib.BLUE, -1)

        line_points.append(image_to_focal_plane(mid_points, O))

    lib.debug_imwrite('opt_points.png', debug)

    # line left-mid and right-mid points on focal plane.
    # axes after transpose: (coord 2, LR 2, line N)
    mid_points_2d = np.array([
        [line[0].left_mid() for line in lines],
        [line[-1].right_mid() for line in lines],
    ]).transpose(2, 0, 1)
    widths = abs(mid_points_2d[0, 1] - mid_points_2d[0, 0])
    mid_points_2d = mid_points_2d[:, :, widths >= 0.9 * np.median(widths)]
    assert mid_points_2d.shape[0:2] == (2, 2)

    # axes (coord 3, LR 2, line N)
    mid_points = image_to_focal_plane(mid_points_2d, O)
    assert mid_points.shape[0:2] == (3, 2)

    line_ts_surface = [newton.t_i_k(R_0, g_0, points, T0) for points in line_points]
    l_m_0 = [Ys.mean() for ts, (_, Ys, _) in line_ts_surface]

    _, (Xs, _, _) = newton.t_i_k(R_0, g_0, mid_points.reshape(3, -1), T0)
    align_0 = Xs.reshape(2, -1).mean(axis=1)  # to LR, N
    print('align_0:', align_0)

    return (np.hstack([theta_0, a_m_0, align_0, l_m_0]),
            (mid_points, line_points))

@lib.timeit
def kim2014(orig):
    lib.debug_prefix = 'dewarp/'

    im = binarize.binarize(orig, algorithm=lambda im: binarize.sauvola(im, k=0.1))
    global bw
    bw = im

    AH, all_lines, lines = get_AH_lines(im)

    im_h, im_w = im.shape
    O = np.array((im_w / 2.0, im_h / 2.0))

    # Test if line start distribution is bimodal.
    line_xs = np.array([line.left() for line in lines])
    bimodal = line_xs.std() / im_w > 0.10

    if bimodal:
        print('Bimodal! Splitting page!')
        bw = im[:, :im_w / 2]
        left = kim2014_individual(orig[:, :im_w / 2], im[:, :im_w / 2], O)
        O_right = O - np.array((im_w / 2, 0))
        bw = im[:, im_w / 2:]
        right = kim2014_individual(orig[:, im_w / 2:], im[:, im_w / 2:], O_right)
        return (left, right)
    else:
        left = kim2014_individual(orig, im, O)
        return (left,)

def kim2014_individual(orig, im, O):
    AH, all_lines, lines = get_AH_lines(im)

    # Estimate viewpoint from vanishing point
    vx, vy = estimate_vanishing(AH, lines) - O

    theta_0 = [atan2(-vy, f) - pi / 2, 0, 0]
    print('theta_0:', theta_0)

    args_0, (mid_points, line_points) = \
        initial_args(lines, O, theta_0=theta_0)

    result = lib.timeit(opt.least_squares)(
        fun=E_0,
        x0=args_0,
        jac=Jac_E_str,
        method='lm',
        args=(mid_points, line_points),
        ftol=1e-3,
        x_scale='jac',
    )
    theta, a_m, _, l_m = unpack_args(result.x)
    print('*** DONE ***')
    print('final norm:', norm(result.fun))
    print('theta:', theta)
    print('a_m:', np.hstack([[0], a_m]))
    print('l_m:', l_m)

    R = R_theta(theta)
    g = Poly(np.hstack([[0], a_m]))

    # line_ts_surface = [newton.t_i_k(R, g, points, T0) for points in line_points]
    # debug_jac(theta, R, g, l_m, line_points, line_ts_surface)
    # debug_plot_g(g, line_ts_surface)

    if lib.debug:
        mesh_2d = make_mesh_2d(all_lines, O, R, g)
        correct_geometry(orig, mesh_2d)

    lib.debug_prefix = 'dewarp2/'

    # im = binarize.binarize(first_pass, algorithm=binarize.ntirogiannis2014)
    # bw = im

    # # find nearest point in new image to original principal point
    # O_distance = norm(mesh_2d - O, axis=2)
    # O = np.array(np.unravel_index(O_distance.argmin(), O_distance.shape))

    # AH, all_lines, lines = get_AH_lines(im)

    # args_0, (mid_points, line_points) = initial_args(lines, O)
    _, (Xs, _, _) = newton.t_i_k(R, g, mid_points.reshape(3, -1), T0)
    align = Xs.reshape(2, -1).mean(axis=1)  # to LR, N

    # TODO: use E_1 if not aligned
    result = lib.timeit(opt.least_squares)(
        fun=E_0,
        x0=np.concatenate([theta, a_m, align, l_m]),
        jac=Jac_E_str,
        method='lm',
        args=(mid_points, line_points),
        ftol=1e-4,
        x_scale='jac',
    )
    theta, a_m, align, l_m = unpack_args(result.x)
    print('*** DONE ***')
    print('final norm:', norm(result.fun))
    print('theta:', theta)
    print('a_m:', np.hstack([[0], a_m]))
    print('l_m:', l_m)
    print('alignment:', (result.fun[-2 * mid_points.shape[-1]:] / LAMBDA_2).astype(int))

    R = R_theta(theta)
    g = Poly(np.hstack([[0], a_m]))

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    for idx, X in enumerate(align):
        line = Line3D.from_point_vec((X, 0, g(X)), (0, 1, 0))
        line.transform(inv(R)).offset(Of).project(FOCAL_PLANE_Z)\
            .offset(-O).draw(debug, color=BLUE if idx == 0 else GREEN)
    lib.debug_imwrite('align.png', debug)

    mesh_2d = make_mesh_2d(all_lines, O, R, g)
    second_pass = correct_geometry(orig, mesh_2d, interpolation=cv2.INTER_LANCZOS4)

    return second_pass

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    lib.debug = True
    out = kim2014(im)
    for i, out_img in enumerate(out):
        cv2.imwrite('dewarped{}.png'.format(i), out_img)

if __name__ == '__main__':
    go(sys.argv)
