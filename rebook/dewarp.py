import cv2
import itertools
# import matplotlib.pyplot as plt
import numpy as np
import skimage.measure
import sys

from math import sqrt, cos, sin, acos, atan2, pi
from numpy.polynomial import Polynomial as P
from scipy import interpolate
from scipy.ndimage import grey_dilation

import algorithm
import binarize
from geometry import Crop, Line
from letters import TextLine
import lib

BLUE = (255, 0, 0)
GREEN = (0, 255, 0)
RED = (0, 0, 255)

# focal length f = 3270.5 pixels
f = 3270.5

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
    bottom_points = np.array(zip(range(width), bottom))
    peak_points = bottom_points[bottom_points[:, 1] == peaks]
    return peak_points

class PolyModel5(object):
    def estimate(self, data):
        self.params = P.fit(data[:, 0], data[:, 1], 5, domain=[-1, 1])
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
            new_model, inliers = skimage.measure.ransac(points, PolyModel5, 10, AH / 15.0)
            out_lines[-1].compress(inliers)
            out_lines[-1].model = new_model
        else:
            out_lines.append(line)

    # debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2RGB)
    # for l in out_lines:
    #     trace_baseline(debug, l, BLUE)
    # lib.debug_imwrite('merged.png', debug)

    print 'original lines:', len(lines), 'merged lines:', len(out_lines)
    return out_lines

@lib.timeit
def remove_outliers(im, AH, lines):
    lines_debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    result = []
    for l in lines:
        if len(l) < 5: continue

        points = np.array([letter.base_point() for letter in l])
        model, inliers = skimage.measure.ransac(points, PolyModel5, 10, AH / 15.0)
        poly = model.params
        l.model = poly
        trace_baseline(lines_debug, l, BLUE)
        for p, is_in in zip(points, inliers):
            color = GREEN if is_in else RED
            cv2.circle(lines_debug, tuple(p.astype(int)), 4, color, -1)

        result.append(TextLine(compress(l, inliers), poly))

    lib.debug_imwrite('lines.png', lines_debug)
    return merge_lines(AH, result)

# x = my + b model weighted t
class LinearXModel(object):
    def estimate(self, data):
        self.params = P.fit(data[:, 1], data[:, 0], 1, domain=[-1, 1])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 1]) - data[:, 0])

def side_lines(im, AH, lines):
    im_h, im_w = im.shape

    left_bounds = np.array([l[0].left_mid() for l in lines])
    right_bounds = np.array([l[-1].right_mid() for l in lines])

    vertical_lines = []
    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for coords in [left_bounds, right_bounds]:
        model, inliers = skimage.measure.ransac(coords, LinearXModel, 3, AH / 10.0)
        vertical_lines.append(model.params)
        for p, inlier in zip(coords, inliers):
            color = GREEN if inlier else RED
            cv2.circle(debug, tuple(p.astype(int)), 4, color, -1)

    for p in vertical_lines:
        cv2.line(debug, (int(p(0)), 0), (int(p(im_h)), im_h), (255, 0, 0), 2)
    lib.debug_imwrite('vertical.png', debug)

    p_left, p_right = vertical_lines
    full_line_mask = np.logical_and(
        abs(p_left(left_bounds[:, 1]) - left_bounds[:, 0]) < AH / 2,
        abs(p_right(right_bounds[:, 1]) - right_bounds[:, 0]) < AH / 2
    )

    return compress(lines, full_line_mask), vertical_lines

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

    C0_left, C0_right = C0[0].left_bot(), C0[-1].right_bot()
    C1_left, C1_right = C1[0].left_bot(), C1[-1].right_bot()

    v_C1_left = Line.from_points(v, C1_left)
    v_C1_right = Line.from_points(v, C1_right)

    C1_left_C0 = v_C1_left.text_line_intersect(C0)
    C1_right_C0 = v_C1_right.text_line_intersect(C0)

    # TODO: fix offsets
    x_min = min(C0_left[0], C1_left_C0[0]) - 20
    x_max = max(C0_right[0], C1_right_C0[0]) + 20
    domain = np.linspace(x_min, x_max, n_points)

    debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    cv2.line(debug, tuple(C0[0].base_point().astype(int)),
             tuple(C0[-1].base_point().astype(int)), GREEN, 2)
    cv2.line(debug, tuple(C1[0].base_point().astype(int)),
             tuple(C1[-1].base_point().astype(int)), GREEN, 2)
    Line.from_points(v, C1_left).draw(debug)
    Line.from_points(v, C1_right).draw(debug)
    lib.debug_imwrite('domain.png', debug)

    return domain, C0, C1

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
    print 'theta:', theta
    A = np.array([
        [1, C[0] / f * -sin(theta)],
        [0, cos(theta) - C[1] / f * sin(theta)]
    ])

    D_points = np.linalg.inv(A).dot(C_points)
    arc_lengths = np.linalg.norm(np.diff(D_points.T, axis=0), axis=1)
    cumulative_arc = np.hstack([[0], np.cumsum(arc_lengths)])
    D = interpolate.interp1d(cumulative_arc, D_points, assume_sorted=True)

    total_arc = cumulative_arc[-1]
    print 'total D arc length:', total_arc
    s_domain = np.linspace(0, total_arc, n_points_w)
    D_points_arc = D(s_domain)
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

def aspect_ratio(im, lines, D, v):
    vx, vy = v
    C0, C1 = C0_C1(lines, v)

    im_h, im_w = im.shape
    # Guess O.
    # TODO: Actually compute this, or crop and keep around, etc.
    O = np.array((im_w / 2.0, im_h / 2.0))

    m = -(vx - O[0]) / (vy - O[1])
    L0 = Line.from_point_slope(C0.first_base(), m)
    L1 = Line.from_point_slope(C1.first_base(), m)
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
    alpha = atan2(np.linalg.norm(p1 - O), f)
    theta = acos(f / sqrt((vx - O[0]) ** 2 + (vy - O[1]) ** 2 + f ** 2))
    beta = pi / 2 - theta

    lp_img = abs(D[0][-1] - D[0][0])
    wp_img = np.linalg.norm(np.diff(D.T, axis=0), axis=1).sum()
    print 'h_img:', h_img, 'l\'_img:', lp_img, 'alpha:', alpha
    print 'l_img:', l_img, 'w\'_img:', wp_img, 'beta:', beta
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
            return l.top_point() + np.array([0, -20])
        else:
            return l.base_point() + np.array([0, 20])

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

    # print longitudes[-1][::20]

    # result = np.array(longitudes)

    # debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    # for l in result:
    #     for p in l:
    #         cv2.circle(debug, tuple(p.astype(long)), 2, BLUE, -1)
    # trace_baseline(debug, C0, RED)
    # trace_baseline(debug, C1, RED)
    # lib.debug_imwrite('mesh.png', debug)

    return np.array(longitudes).transpose(1, 0, 2)

@lib.timeit
def correct_geometry(orig, mesh, r):
    # coordinates (u, v) on mesh -> mesh[u][v] = (x, y) in distorted image
    mesh32 = mesh.astype(np.float32)
    xmesh, ymesh = mesh32[:, :, 0], mesh32[:, :, 1]
    conv_xmesh, conv_ymesh = cv2.convertMaps(xmesh, ymesh, cv2.CV_16SC2)
    out = cv2.remap(orig, conv_xmesh, conv_ymesh, interpolation=cv2.INTER_LINEAR)
    lib.debug_imwrite('dewarped.png', out)

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

def get_AH_lines(im):
    AH = algorithm.dominant_char_height(im)
    print 'AH =', AH
    letters = algorithm.letter_contours(AH, im)
    print 'collating...'
    all_lines = algorithm.collate_lines_2(AH, letters)
    all_lines.sort(key=lambda l: l[0].y)

    if lib.debug:
        global curvature_debug
        curvature_debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
        all_lines = filter(valid_curvature, all_lines)
        lib.debug_imwrite('curvature.png', curvature_debug)

    lines = remove_outliers(im, AH, all_lines)

    return AH, all_lines, lines

def dewarp(orig):
    # Meng et al., Metric Rectification of Curved Document Images
    lib.debug = True
    im = binarize.binarize(orig, algorithm=lambda im: binarize.yan(im, alpha=0.3))
    global bw
    bw = im
    im_h, im_w = im.shape

    AH, all_lines, lines = get_AH_lines(im)

    # debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    # for l in all_lines:
    #     for l1, l2 in zip(l, l[1:]):
    #         cv2.line(debug, tuple(l1.base_point().astype(int)),
    #                  tuple(l2.base_point().astype(int)), BLUE, 2)
    # lib.debug_imwrite('all_lines.png', debug)

    lines, verticals = side_lines(im, AH, lines)
    print 'full lines:', len(lines)

    p_left, p_right = verticals
    vy, = (p_left - p_right).roots()
    vx = p_left(vy)
    v0 = np.array((vx, vy))

    # TODO: make vanishing point determination optimal
    v = v0
    print 'vanishing point:', v

    box = min_crop(all_lines)
    D, C_arc = estimate_directrix(lines, v, box.w)

    r = aspect_ratio(im, lines, D, v)
    r = 1.7  # TODO: fix
    print 'aspect ratio H/W:', r

    print 'generating mesh...'
    mesh = generate_mesh(all_lines, lines, C_arc, v, r * box.w)

    print 'dewarping...'
    dewarped = correct_geometry(orig, mesh, r)

    # print 'binarizing...'
    # dewarped_bw = binarize.binarize(dewarped, algorithm=lambda im: binarize.yan(im, alpha=0.3))

    # print 'fine dewarping...'
    # fine = dewarp_fine(dewarped_bw)

    return dewarped

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    out = dewarp(im)
    cv2.imwrite('out.png', out)

if __name__ == '__main__':
    go(sys.argv)
