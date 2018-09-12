from __future__ import print_function

import cv2
import itertools
import numpy as np
import sys

from math import atan2, pi
from numpy import dot, newaxis
from numpy.linalg import norm, inv, solve
from numpy.polynomial import Polynomial as Poly
from scipy import optimize as opt
from scipy import interpolate
from scipy.linalg import block_diag
from skimage.measure import ransac

import algorithm
import binarize
import collate
import crop
from geometry import Crop
import lib
from lib import RED, GREEN, BLUE, draw_circle, draw_line
import newton

# focal length f = 3270.5 pixels
f = 3270.5
Of = np.array([0, 0, f], dtype=np.float64)

def compress(l, flags):
    return list(itertools.compress(l, flags))

def arc_length_points(xs, ys, n_points):
    arc_points = np.stack((xs, ys))
    arc_lengths = norm(np.diff(arc_points, axis=1), axis=0)
    cumulative_arc = np.hstack([[0], np.cumsum(arc_lengths)])
    D = interpolate.interp1d(cumulative_arc, arc_points, assume_sorted=True)

    total_arc = cumulative_arc[-1]
    if lib.debug: print('total D arc length:', total_arc)
    s_domain = np.linspace(0, total_arc, n_points)
    return D(s_domain), total_arc

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
            draw_circle(debug, p, 4, color=GREEN if inlier else RED)

    for p in vertical_lines:
        draw_line(debug, (p(0), 0), (p(im_h), im_h), BLUE, 2)
    lib.debug_imwrite('vertical.png', debug)

    return vertical_lines

def estimate_vanishing(AH, lines):
    p_left, p_right = side_lines(AH, lines)
    vy, = (p_left - p_right).roots()
    return np.array((p_left(vy), vy))

class PolyModel5(object):
    def estimate(self, data):
        self.params = Poly.fit(data[:, 0], data[:, 1], 5, domain=[-1, 1])
        return True

    def residuals(self, data):
        return abs(self.params(data[:, 0]) - data[:, 1])

def trace_baseline(im, line, color=BLUE):
    domain = np.linspace(line.left() - 100, line.right() + 100, 200)
    points = np.vstack([domain, line.model(domain)]).T
    for p1, p2 in zip(points, points[1:]):
        draw_line(im, p1, p2, color=color, thickness=1)

def merge_lines(AH, lines):
    if len(lines) == 0: return lines

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

# @lib.timeit
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
            draw_circle(debug, p, 4, color=color)

        l.compress(inliers)
        result.append(l)

    for l in result:
        draw_circle(debug, l[0].left_mid(), 6, BLUE, -1)
        draw_circle(debug, l[-1].right_mid(), 6, BLUE, -1)

    lib.debug_imwrite('lines.png', debug)
    return merge_lines(AH, result)

# @lib.timeit
def correct_geometry(orig, mesh, interpolation=cv2.INTER_LINEAR):
    # coordinates (u, v) on mesh -> mesh[u][v] = (x, y) in distorted image
    mesh32 = mesh.astype(np.float32)
    xmesh, ymesh = mesh32[:, :, 0], mesh32[:, :, 1]
    conv_xmesh, conv_ymesh = cv2.convertMaps(xmesh, ymesh, cv2.CV_16SC2)
    out = cv2.remap(orig, conv_xmesh, conv_ymesh, interpolation=interpolation,
                    borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    lib.debug_imwrite('corrected.png', out)

    return out

def get_AH_lines(im):
    all_letters = algorithm.all_letters(im)
    AH = algorithm.dominant_char_height(im, letters=all_letters)
    if lib.debug: print('AH =', AH)
    letters = algorithm.filter_size(AH, im, letters=all_letters)
    all_lines = collate.collate_lines(AH, letters)
    all_lines.sort(key=lambda l: l[0].y)

    combined = algorithm.combine_underlined(AH, im, all_lines, all_letters)

    filtered = algorithm.remove_stroke_outliers(bw, combined, k=2.0)
    filtered = algorithm.filter_width_deviation(bw, filtered)

    lines = remove_outliers(im, AH, filtered)
    # lines = combined

    # if lib.debug:
    #     debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
    #     for l in all_lines:
    #         for l1, l2 in zip(l, l[1:]):
    #             cv2.line(debug, tuple(l1.base_point().astype(int)),
    #                     tuple(l2.base_point().astype(int)), RED, 2)
    #     lib.debug_imwrite('all_lines.png', debug)

    return AH, lines, all_lines

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
def line_base_points_modeled(line, O):
    model = line.fit_poly()
    x0, _ = line[0].base_point() + 5
    x1, _ = line[-1].base_point() - 5
    domain = np.linspace(x0, x1, len(line))
    points = np.stack([domain, model(domain)])
    return image_to_focal_plane(points, O)

def line_base_points(line, O):
    return image_to_focal_plane(line.base_points().T, O)

# represents g(x) = 1/w h(wx)
class NormPoly(object):
    def __init__(self, coef, omega):
        self.h = Poly(coef)
        self.omega = omega

    def __call__(self, x):
        return self.h(self.omega * x) / self.omega

    def deriv(self):
        return NormPoly(self.omega * self.h.deriv().coef, self.omega)

    def degree(self):
        return self.h.degree()

    def split(self):
        return False

    @property
    def coef(self):
        return self.h.coef

class SplitPoly(object):
    def __init__(self, T, left, right):
        self.T = T
        self.left = left
        self.right = right

    def __call__(self, x):
        T = self.T
        if np.isscalar(x):
            return self.left(x - T) if x < T else self.right(x - T)
        else:
            return np.where(x < T, self.left(x - T), self.right(x - T))

    def deriv(self):
        return SplitPoly(self.T, self.left.deriv(), self.right.deriv())

    def degree(self):
        return max(self.left.degree(), self.right.degree())

    def split(self):
        return True

def split_lengths(array, lengths):
    return np.split(array, np.cumsum(lengths))

DEGREE = 13
OMEGA = 1e-1
def unpack_args(args, n_pages):
    # theta: 3; a_m: DEGREE; align: 2; l_m: len(lines)
    theta, a_m_all, align_all, (T,), l_m = \
        split_lengths(np.array(args), (3, DEGREE * n_pages, 2 * n_pages, 1))
    T = 0
    # theta[1] = 0.

    a_ms = np.split(a_m_all, n_pages)
    aligns = align_all.reshape(n_pages, -1)

    if n_pages == 1:
        g = NormPoly(np.concatenate([[0], a_ms[0]]), OMEGA)
    elif n_pages == 2:
        g = SplitPoly(T,
                      NormPoly(np.concatenate([[0], a_ms[0]]), OMEGA),
                      NormPoly(np.concatenate([[0], a_ms[1]]), OMEGA))

    return theta, a_ms, aligns, T, l_m, g

E_str_t0s = []
def E_str_project(R, g, base_points, t0s_idx):
    global E_str_t0s
    if len(E_str_t0s) <= t0s_idx:
        E_str_t0s.extend([None] * (t0s_idx - len(E_str_t0s) + 1))
    if E_str_t0s[t0s_idx] is None:
        E_str_t0s[t0s_idx] = \
            [np.full((points.shape[1],), np.inf) for points in base_points]

    # print([point.shape for point in base_points])
    # print([t0s.shape for t0s in E_str_t0s])

    return [newton.t_i_k(R, g, points, t0s) \
            for points, t0s in zip(base_points, E_str_t0s[t0s_idx])]

class Loss(object):
    def __add__(self, other):
        return AddLoss(self, other)

    def __mul__(self, other):
        return MulLoss(self, other)

    def gradient(self, x, *args):
        return self.jac(x, *args).dot(self.residuals(x, *args))

class AddLoss(Loss):
    def __init__(self, a, b):
        self.a, self.b = a, b

    def residuals(self, x, *args):
        return np.concatenate([self.a.residuals(x, *args),
                               self.b.residuals(x, *args)])

    def jac(self, x, *args):
        a_jac = self.a.jac(x, *args)
        b_jac = self.b.jac(x, *args)
        return np.concatenate((a_jac, b_jac))

class MulLoss(Loss):
    def __init__(self, inner, c):
        self.inner = inner
        self.c = c

    def residuals(self, x, *args):
        return self.c * self.inner.residuals(x, *args)

    def jac(self, x, *args):
        return self.c * self.inner.jac(x, *args)

class DebugLoss(Loss):
    def __init__(self, inner):
        self.inner = inner

    def residuals(self, *args):
        result = self.inner.residuals(*args)
        if lib.debug: print('norm: {:3.6f}'.format(norm(result)))
        return result

    def jac(self, *args):
        return self.inner.jac(*args)

class Preproject(Loss):
    def __init__(self, inner, base_points, n_pages):
        self.inner = inner
        self.base_points = base_points
        self.n_pages = n_pages
        self.last_x = None
        self.last_projection = None

    def project(self, args):
        theta, _, _, T, l_m, g = unpack_args(args, self.n_pages)
        # print '    theta:', theta
        # print '    a_m:', g.coef

        if self.last_x is None or self.last_x.shape != args.shape or np.any(args != self.last_x):
            R = R_theta(theta)
            self.last_x = args
            self.last_projection = E_str_project(R, g, self.base_points, 0)

        return self.last_projection

    def residuals(self, args):
        all_ts_surface = self.project(args)
        _, _, _, T, _, _ = unpack_args(args, self.n_pages)
        return self.inner.residuals(args, all_ts_surface)

    def jac(self, args):
        all_ts_surface = self.project(args)
        return self.inner.jac(args, all_ts_surface)

class Regularize_T(Loss):
    def __init__(self, base_points, n_pages):
        self.base_points = base_points
        self.n_pages = n_pages

    def residuals(self, args, line_ts_surface):
        residuals = [ts + 1 for ts, _ in line_ts_surface]
        return np.concatenate(residuals)

    def jac(self, args, line_ts_surface):
        theta, a_m, _, T, l_m, g = unpack_args(args, self.n_pages)
        R = R_theta(theta)
        dR = dR_dtheta(theta, R)

        gp = g.deriv()

        all_points = np.concatenate(self.base_points, axis=1)
        all_ts = np.concatenate([ts for ts, _ in line_ts_surface])
        all_surface = np.concatenate([surface for _, surface in line_ts_surface],
                                     axis=1)

        dtheta = dti_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface).T
        # dtheta[:, 1] = 0

        return np.concatenate((
            dtheta,
            dti_dam(R, g, gp, all_points, all_ts, all_surface).T,
            np.zeros((all_ts.shape[0], 2 * self.n_pages + 1 + len(self.base_points)),
                     dtype=np.float64),
        ), axis=1)

OUTER_LINE_WEIGHT = 2
def line_weights(points):
    return 1 + np.abs(np.linspace(-OUTER_LINE_WEIGHT + 1, OUTER_LINE_WEIGHT - 1, points.shape[-1]))

class E_str(Loss):
    def __init__(self, base_points, n_pages, weight_outer=True, scale_t=False):
        self.base_points = base_points
        self.all_points = np.concatenate(base_points, axis=1)
        self.all_weights = np.concatenate([line_weights(line) for line in self.base_points])
        self.n_pages = n_pages
        self.weight_outer = weight_outer  # Weight outer letters in line more heavily
        self.scale_t = scale_t  # Scale by - 1 / t

    # l_m = fake parameter representing line position
    # base_points = text base points on focal plane
    @staticmethod
    def unpacked(all_ts_surface, l_m):
        assert len(all_ts_surface) == l_m.shape[0]

        residuals = [Ys - l_k for (_, (_, Ys, _)), l_k in zip(all_ts_surface, l_m)]
        return np.concatenate(residuals)

    def residuals(self, args, all_ts_surface):
        theta, _, _, T, l_m, g = unpack_args(args, self.n_pages)
        result = E_str.unpacked(all_ts_surface, l_m)

        if self.weight_outer:
            result *= self.all_weights

        if self.scale_t:
            all_ts = np.concatenate([ts for ts, _ in all_ts_surface])
            result /= -all_ts

        return result

    def jac(self, args, all_ts_surface):
        theta, a_m, _, T, l_m, g = unpack_args(args, self.n_pages)
        R = R_theta(theta)
        dR = dR_dtheta(theta, R)

        gp = g.deriv()

        all_ts = np.concatenate([ts for ts, _ in all_ts_surface])
        all_surface = np.concatenate([surface for _, surface in all_ts_surface],
                                     axis=1)
        residuals = E_str.unpacked(all_ts_surface, l_m)

        dtheta = dE_str_dtheta(theta, R, dR, g, gp, self.all_points, all_ts, all_surface)
        dam = dE_str_dam(R, g, gp, self.all_points, all_ts, all_surface)
        # dtheta[:, 1] = 0

        if self.scale_t:
            dtheta -= residuals[:, newaxis] / all_ts[:, newaxis] * dti_dtheta(theta, R, dR, g, gp, self.all_points, all_ts, all_surface).T
            dam -= residuals[:, newaxis] / all_ts[:, newaxis] * dti_dam(R, g, gp, self.all_points, all_ts, all_surface).T

        result = np.concatenate((
            dtheta,
            dam,
            # Doesn't depend on alignment:
            np.zeros((all_ts.shape[0], 2 * self.n_pages), dtype=np.float64),
            dE_str_dT(R, g, gp, self.all_points, all_ts, all_surface),
            dE_str_dl_k(self.base_points),
        ), axis=1)

        if self.weight_outer:
            result *= self.all_weights[:, newaxis]

        if self.scale_t:
            result /= -all_ts[:, newaxis]

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

def dti_dam(R, g, gp, all_points, all_ts, all_surface):
    R1, R2, R3 = R

    Xs, _, _ = all_surface

    denom = R3.dot(all_points) - gp(Xs) * R1.dot(all_points)
    if isinstance(g, SplitPoly):
        powers = np.vstack([(Xs - g.T) ** m * g.left.omega ** (m - 1)
                            for m in range(1, DEGREE + 1)])
        ratio = powers / denom
        # print('on left:', np.count_nonzero(Xs <= g.T))
        # print('on right:', np.count_nonzero(Xs > g.T))
        left_block = np.where(Xs <= g.T, ratio, 0)
        right_block = np.where(Xs > g.T, ratio, 0)
        return np.concatenate([left_block, right_block])
    else:
        powers = np.vstack([Xs ** m * g.omega ** (m - 1)
                            for m in range(1, DEGREE + 1)])
        return powers / denom

def dE_str_dam(R, g, gp, all_points, all_ts, all_surface):
    R1, R2, R3 = R

    dt = dti_dam(R, g, gp, all_points, all_ts, all_surface)

    return (R2.dot(all_points) * dt).T

def dE_str_dl_k(base_points):
    blocks = [np.full((l.shape[-1], 1), -1) for l in base_points]
    return block_diag(*blocks)

def dE_str_dT(R, g, gp, all_points, all_ts, all_surface):
    R1, R2, R3 = R

    Xs, _, _, = all_surface
    gp_val = gp(Xs)

    num = R2.dot(all_points) * gp_val
    denom = R1.dot(all_points) * gp_val - R3.dot(all_points)
    result = (num / denom)[:, newaxis]

    return np.zeros(result.shape, dtype=np.float64)
    return result

def debug_plot_g(g, line_ts_surface):
    import matplotlib.pyplot as plt
    all_points_XYZ = np.concatenate([points for _, points in line_ts_surface],
                                    axis=1)
    domain = np.linspace(all_points_XYZ[0].min(), all_points_XYZ[0].max(), 100)
    plt.plot(domain, g(domain))
    # domain = np.linspace(-im_w / 2, im_w / 2, 100)
    # plt.plot(domain, g(domain))
    plt.show()

def debug_jac(theta, R, g, l_m, base_points, line_ts_surface):
    dR = dR_dtheta(theta, R)
    gp = g.deriv()

    all_points = np.concatenate(base_points, axis=1)
    all_ts = np.concatenate([ts for ts, _ in line_ts_surface])
    all_surface = np.concatenate([surface for _, surface in line_ts_surface], axis=1)

    print('dE_str_dtheta')
    print(dE_str_dtheta(theta, R, dR, g, gp, all_points, all_ts, all_surface).T)
    for i in range(3):
        delta = np.zeros(3)
        inc = norm(theta) / 4096
        delta[i] = inc
        diff = E_str(theta + delta, g, l_m, base_points) - E_str(theta - delta, g, l_m, base_points)
        print(diff / (2 * inc))

    print()

    print('dE_str_dam')
    analytical = dE_str_dam(R, g, g.deriv(), all_points, all_ts, all_surface).T
    gl = g.left
    gr = g.right
    print('==== LEFT ====')
    for i in range(1, DEGREE + 1):
        delta = np.zeros(DEGREE + 1)
        inc = gl.coef[i] / 4096
        delta[i] = inc
        diff = E_str(theta, SplitPoly(g.T, NormPoly(gl.coef + delta, gl.omega), gr), l_m, base_points) \
            - E_str(theta, SplitPoly(g.T, NormPoly(gl.coef - delta, gl.omega), gr), l_m, base_points)
        nonzero = np.logical_or(
            abs(analytical[i - 1]) > 1e-7,
            abs(diff / (2 * inc)) > 1e-7,
        )
        print('X  ', all_surface[0, nonzero])
        print('ana', analytical[i - 1][nonzero])
        print('dif', (diff / (2 * inc))[nonzero])
        print()

    print('==== RIGHT ====')
    for i in range(1, DEGREE + 1):
        delta = np.zeros(DEGREE + 1)
        inc = gr.coef[i] / 4096
        delta[i] = inc
        diff = E_str(theta, SplitPoly(g.T, gl, NormPoly(gr.coef + delta, gr.omega)), l_m, base_points) \
            - E_str(theta, SplitPoly(g.T, gl, NormPoly(gr.coef - delta, gr.omega)), l_m, base_points)
        nonzero = np.logical_or(
            abs(analytical[DEGREE + i - 1]) > 1e-7,
            abs(diff / (2 * inc)) > 1e-7,
        )
        print('X  ', all_surface[0, nonzero])
        print('ana', analytical[DEGREE + i - 1][nonzero])
        print('dif', (diff / (2 * inc))[nonzero])
        print()

    if g.split():
        print('dE_str_dT (T = {:.3f})'.format(g.T))
        print(dE_str_dT(R, g, gp, all_points, all_ts, all_surface).T)
        inc = 1e-2
        diff = E_str(theta, SplitPoly(g.T + inc, g.left, g.right), l_m, base_points) \
            - E_str(theta, SplitPoly(g.T - inc, g.left, g.right), l_m, base_points)
        print(diff / (2 * inc))

E_align_t0s = []
def E_align_project(R, g, all_points, t0s_idx):
    global E_align_t0s
    if len(E_align_t0s) <= t0s_idx:
        E_align_t0s.extend([None] * (t0s_idx - len(E_align_t0s) + 1))
    if E_align_t0s[t0s_idx] is None:
        E_align_t0s[t0s_idx] = np.full((all_points.shape[1],), np.inf)

    return newton.t_i_k(R, g, all_points, E_align_t0s[t0s_idx])

class E_align_page(Loss):
    def __init__(self, side_points, left, right, n_pages, page_index, n_total_lines):
        assert left or right
        self.left, self.right = left, right
        self.n_pages = n_pages
        self.page_index = page_index
        self.n_total_lines = n_total_lines

        # [coord, side, line]
        self.side_points = side_points[:, self.side_slice(), :]
        self.all_side_points = self.side_points.reshape(3, -1)

    def side_slice(self):
        if self.left and self.right:
            return np.s_[:]
        elif self.left:
            return np.s_[:1]
        else:
            return np.s_[1:]

    def n_align(self):
        return int(self.left) + int(self.right)

    def E_align(self, theta, g, align):
        R = R_theta(theta)

        _, (Xs, _, _) = E_align_project(R, g, self.all_side_points, 0)
        Xs.shape = (self.n_align(), -1)  # [side, line]

        return (Xs - align[self.side_slice()][:, newaxis]).flatten()

    def residuals(self, args):
        theta, _, align_all, T, _, g = unpack_args(args, self.n_pages)
        return self.E_align(theta, g, align_all[self.page_index])

    def dE_align_dam(self, theta, R, g, gp, all_ts, all_surface):
        R1, _, _ = R

        dt = dti_dam(R, g, gp, self.all_side_points, all_ts, all_surface)

        return (R1.dot(self.all_side_points) * dt).T

    def dE_align_dtheta(self, theta, R, dR, g, gp, all_ts, all_surface):
        R1, _, _ = R
        dR1 = dR[:, 0]
        dR13 = dR[:, 0, 2]

        dt = dti_dtheta(theta, R, dR, g, gp, self.all_side_points, all_ts, all_surface)

        term1 = dR1.dot(self.all_side_points) * all_ts
        term2 = R1.dot(self.all_side_points) * dt
        term3 = -dR13 * f

        return term1.T + term2.T + term3

    def dE_align_dalign(self):
        N_lines = self.side_points.shape[-1]
        result = np.zeros((N_lines * self.n_align(), self.n_pages * 2),
                          dtype=np.float64)

        block = []
        if self.left:
            block.append([-1, 0])
        if self.right:
            block.append([0, -1])

        column = self.page_index * 2
        result[:, column:column + 2] = np.repeat(block, N_lines, axis=0)

        return result

    def dE_align_dT(self, R, g, gp, all_ts, all_surface):
        R1, R2, R3 = R

        Xs, _, _, = all_surface
        gp_val = gp(Xs)

        num = R1.dot(self.all_side_points) * gp_val
        denom = R1.dot(self.all_side_points) * gp_val - R3.dot(self.all_side_points)
        result = (num / denom)[:, newaxis]

        return np.zeros(result.shape, dtype=np.float64)
        return result

    def jac(self, args):
        theta, a_m, _, _, _, g = unpack_args(args, self.n_pages)
        R = R_theta(theta)
        dR = dR_dtheta(theta, R)
        gp = g.deriv()

        N_residuals = self.all_side_points.shape[-1]

        all_ts, all_surface = E_align_project(R, g, self.all_side_points, 0)

        return np.concatenate((
            self.dE_align_dtheta(theta, R, dR, g, gp, all_ts, all_surface),
            self.dE_align_dam(theta, R, g, gp, all_ts, all_surface),
            self.dE_align_dalign(),
            self.dE_align_dT(R, g, gp, all_ts, all_surface),
            np.zeros((N_residuals, self.n_total_lines), dtype=np.float64)  # dl_k
        ), axis=1)

def make_E_align_page(page, AH, O, n_pages, page_index, n_total_lines):
    # line left-mid and right-mid points on focal plane.
    # (LR 2, line N, coord 2)
    side_points_2d = np.array([
        [line[0].left_mid() for line in page],
        [line[-1].right_mid() for line in page],
    ])

    side_inliers = [ransac(coords, LinearXModel, 3, AH / 5.0)[1] for coords in side_points_2d]
    inliers = np.logical_and(side_inliers[0], side_inliers[1])

    # axes after transpose: (coord 2, LR 2, line N)
    side_points_2d_filtered = side_points_2d[:, inliers].transpose(2, 0, 1)

    # widths = abs(side_points_2d[0, 1] - side_points_2d[0, 0])
    # side_points_2d = side_points_2d[:, :, widths >= 0.9 * np.median(widths)]
    assert side_points_2d_filtered.shape[0:2] == (2, 2)

    # axes (coord 3, LR 2, line N)
    side_points = image_to_focal_plane(side_points_2d_filtered, O)
    assert side_points.shape[0:2] == (3, 2)

    return E_align_page(side_points, True, True, n_pages, page_index, n_total_lines)

def make_E_align(pages, AH, O):
    n_pages = len(pages)
    n_total_lines = sum((len(page) for page in pages)) + \
        sum((sum((len(line.underlines) for line in page)) for page in pages))
    losses = [
        make_E_align_page(page, AH, O, n_pages, i, n_total_lines) \
        for i, page in enumerate(pages)
    ]
    return reduce(lambda a, b: a + b, losses)

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
            draw_circle(debug, p, color=color)
        lib.debug_imwrite(filename, debug)

# @lib.timeit
def make_mesh_2d(all_lines, O, R, g, n_points_w=None):
    all_letters = np.concatenate([line.letters for line in all_lines])
    corners_2d = np.concatenate([letter.corners() for letter in all_letters]).T
    corners = image_to_focal_plane(corners_2d, O)
    t0s = np.full((corners.shape[1],), np.inf, dtype=np.float64)
    corners_t, corners_XYZ = newton.t_i_k(R, g, corners, t0s)

    corners_X, _, corners_Z = corners_XYZ
    relative_Z_error = np.abs(g(corners_X) - corners_Z) / corners_Z
    corners_XYZ = corners_XYZ[:, np.logical_and(relative_Z_error <= 0.02,
                                                abs(corners_Z) < 1e6,
                                                corners_t < 0)]
    corners_X, _, _ = corners_XYZ

    debug_print_points('corners.png', corners_2d)

    if lib.debug:
        try:
            import matplotlib.pyplot as plt
            ax = plt.axes()
            box_XY = Crop.from_points(corners_XYZ[:2]).expand(0.01)
            x_min, y_min, x_max, y_max = box_XY

            for y in np.linspace(y_min, y_max, 3):
                xs = np.linspace(x_min, x_max, 200)
                ys = np.full(200, y)
                zs = g(xs)
                points = np.stack([xs, ys, zs])
                points_r = inv(R).dot(points) + Of[:, newaxis]
                ax.plot(points_r[0], points_r[2])

            base_xs = np.array([corners[0].min(), corners[0].max()])
            base_zs = np.array([-3270.5, -3270.5])
            ax.plot(base_xs, base_zs)
            ax.set_aspect('equal')
            plt.savefig('dewarp/camera.png')
        except Exception as e:
            print(e)
            import IPython
            IPython.embed()

    if g.split():
        meshes = [
            make_mesh_2d_indiv(all_lines, corners_XYZ[:, corners_X <= g.T], O, R, g, n_points_w=n_points_w),
            make_mesh_2d_indiv(all_lines, corners_XYZ[:, corners_X > g.T], O, R, g, n_points_w=n_points_w),
        ]
    else:
        meshes = [make_mesh_2d_indiv(all_lines, corners_XYZ, O, R, g, n_points_w=n_points_w)]

    for i, mesh in enumerate(meshes):
        # debug_print_points('mesh{}.png'.format(i), mesh, step=20)
        pass

    return meshes

def make_mesh_2d_indiv(all_lines, corners_XYZ, O, R, g, n_points_w=None):
    box_XYZ = Crop.from_points(corners_XYZ[:2]).expand(0.01)
    if lib.debug: print('box_XYZ:', box_XYZ)

    if n_points_w is None:
        # 90th percentile line width a good guess
        n_points_w = 1.2 * np.percentile(np.array([line.width() for line in all_lines]), 90)
    mesh_XYZ_x = np.linspace(box_XYZ.x0, box_XYZ.x1, 400)
    mesh_XYZ_z = g(mesh_XYZ_x)
    mesh_XYZ_xz_arc, total_arc = arc_length_points(mesh_XYZ_x, mesh_XYZ_z,
                                                   int(n_points_w))
    mesh_XYZ_x_arc, _ = mesh_XYZ_xz_arc

    # TODO: think more about estimation of aspect ratio for mesh
    n_points_h = n_points_w * box_XYZ.h / total_arc
    # n_points_h = n_points_w * 1.7

    mesh_XYZ_y = np.linspace(box_XYZ.y0, box_XYZ.y1, n_points_h)
    mesh_XYZ = make_mesh_XYZ(mesh_XYZ_x_arc, mesh_XYZ_y, g)
    mesh_2d = gcs_to_image(mesh_XYZ, O, R)
    if lib.debug: print('mesh:', Crop.from_points(mesh_2d))

    # make sure meshes are not reversed
    if mesh_2d[0, :, 0].mean() > mesh_2d[0, :, -1].mean():
        mesh_2d = mesh_2d[:, :, ::-1]

    if mesh_2d[1, 0].mean() > mesh_2d[1, -1].mean():
        mesh_2d = mesh_2d[:, ::-1, :]

    return mesh_2d.transpose(1, 2, 0)

def lm(fun, x0, jac, args=(), kwargs={}, ftol=1e-6, max_nfev=10000, x_scale=None,
       geodesic_accel=False, uphill_steps=False):
    LAM_UP = 1.5
    LAM_DOWN = 5.

    if x_scale is None:
        x_scale = np.ones(x0.shape[0], dtype=np.float64)

    x = x0
    xs = x / x_scale
    lam = 100.

    r = fun(x, *args, **kwargs)
    C = dot(r, r) / 2
    Js = jac(x, *args, **kwargs) * x_scale[newaxis, :]
    dC = dot(Js.T, r)
    JsTJs = dot(Js.T, Js)
    assert r.shape[0] == Js.shape[0]

    I = np.eye(Js.shape[1])

    for step in range(max_nfev):
        xs_new = xs - solve(JsTJs + lam * I, dC)
        x_new = xs_new * x_scale

        r_new = fun(x_new, *args, **kwargs)
        C_new = dot(r_new, r_new) / 2
        print('trying step: size {:.3g}, C {:.3g}, lam {:.3g}'.format(
            norm(x - x_new), C_new, lam
        ))
        # print(x - x_new)
        if C_new >= C:
            lam *= LAM_UP
            if lam >= 1e6: break
            continue

        relative_err = abs(C - C_new) / C
        if relative_err <= ftol:
            break

        xs = xs_new
        print(xs)
        x = xs * x_scale
        r = r_new

        C = C_new

        if C < 1e-6: break

        Js = jac(x, *args, **kwargs) * x_scale[newaxis, :]
        dC = dot(Js.T, r)
        JsTJs = dot(Js.T, Js)
        lam /= LAM_DOWN

    return opt.OptimizeResult(x=x, fun=r)


def Jac_to_grad_lsq(residuals, jac, x, args):
    jacobian = jac(x, *args)
    return residuals.dot(jacobian)

def lsq(func, jac, x_scale):
    def result(xs, *args):
        residuals = func(xs * x_scale, *args)
        return np.dot(residuals, residuals), \
            Jac_to_grad_lsq(residuals, jac, xs * x_scale, args) * x_scale

    return result

def kim2014(orig, O=None, split=True, n_points_w=None):
    im = binarize.binarize(orig, algorithm=lambda im: binarize.sauvola_noisy(im, k=0.1))
    global bw
    bw = im

    im_h, im_w = im.shape

    AH, lines, _ = get_AH_lines(im)

    if O is None:
        O = np.array((im_w / 2.0, im_h / 2.0))

    if split:
        # Test if line start distribution is bimodal.
        line_xs = np.array([line.left() for line in lines])
        bimodal = line_xs.std() / im_w > 0.10
        dual = bimodal and im_w > im_h
    else:
        dual = False

    if dual:
        print('Bimodal! Splitting page!')
        pages = crop.split_lines(lines)

        n_points_w = 1.2 * np.percentile(np.array([line.width() for line in lines]), 90)

        if lib.debug:
            debug = cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)
            for page in pages:
                page_crop = Crop.from_lines(page).expand(0.005)
                # print(page_crop)
                page_crop.draw(debug)
            lib.debug_imwrite('split.png', debug)

        page_crops = [Crop.from_lines(page) for page in pages]
        if len(page_crops) == 2:
            [c0, c1] = page_crops
            split_x = (c0.x1 + c1.x0) / 2
            page_crops = [
                c0.union(Crop(0, 0, split_x, im_h)),
                c1.union(Crop(split_x, 0, im_w, im_h))
            ]

        result = []
        for i, (page, page_crop) in enumerate(zip(pages, page_crops)):
            print('==== PAGE {} ===='.format(i))
            lib.debug_prefix.append('page{}'.format(i))

            page_image = page_crop.apply(orig)
            page_bw = page_crop.apply(im)
            page_AH, page_lines, _ = get_AH_lines(page_bw)
            new_O = O - np.array((page_crop.x0, page_crop.y0))
            lib.debug_imwrite('precrop.png', im)
            lib.debug_imwrite('page.png', page_image)

            dewarper = Kim2014(page_image, page_bw, page_lines, [page_lines],
                               new_O, page_AH, n_points_w)
            result.append(dewarper.run_retry()[0])

            lib.debug_prefix.pop()

        return result
    else:
        dewarper = Kim2014(orig, im, lines, [lines], O, AH, n_points_w)
        return dewarper.run_retry()

class Kim2014(object):
    def __init__(self, orig, im, lines, pages, O, AH, n_points_w):
        self.orig = orig
        self.im = im
        self.lines = lines
        self.pages = pages
        self.O = O
        self.AH = AH
        self.n_points_w = n_points_w

        for page in self.pages:
            page.sort(key=lambda l: l[0].y)

        # line points on focal plane
        self.base_points = [line_base_points(line, O) for line in lines]
        # make underlines straight as well
        for line in lines:
            # if line.underlines: print('underlines:', len(line.underlines))
            for underline in line.underlines:
                mid_contour = (underline.top_contour() + underline.bottom_contour()) / 2
                all_mid_points = np.stack([
                    underline.x + np.arange(underline.w), mid_contour,
                ])
                mid_points = all_mid_points[:, :]

                self.base_points.append(image_to_focal_plane(mid_points, O))

    def initial_args(self):
        # Estimate viewpoint from vanishing point
        vanishing_points = [estimate_vanishing(self.AH, page) \
                            for page in self.pages]
        mean_image_vanishing = np.mean(vanishing_points, axis=0)
        vanishing = np.concatenate([mean_image_vanishing - self.O, [-f]])
        vx, vy, _ = vanishing
        if lib.debug: print(' v:', vanishing)

        xz_ratio = -f / vx  # theta_x / theta_z
        norm_theta_sq = (atan2(np.sqrt(vx ** 2 + f ** 2), vy) - pi) ** 2
        theta_z = np.sqrt(norm_theta_sq / (xz_ratio ** 2 + 1))
        theta_x = xz_ratio * theta_z
        theta_x

        # theta_0 = np.array([theta_x, 0, theta_z])
        # print('theta_0:', theta_0)
        # print('theta_0 dot ey:', theta_0.dot(np.array([0, 1, 0])))
        # print('theta_0 dot v:', theta_0.dot(vanishing))
        # theta_0 = np.array([0.1, 0, 0], dtype=np.float64)
        theta_0 = (np.random.rand(3) - 0.5) / 4

        # flat surface as initial guess.
        # NB: coeff 0 forced to 0 here. not included in opt.
        a_m_0 = [0] * (DEGREE * len(self.pages))

        R_0 = R_theta(theta_0)
        _, ROf_y, ROf_z = R_0.dot(Of)
        if lib.debug: print('Rv:', R_0.dot(np.array((vx, vy, -f))))

        all_surface = [R_0.dot(-points - Of[:, newaxis]) for points in self.base_points]
        l_m_0 = [Ys.mean() for _, Ys, _ in all_surface]

        align_0 = [-1000, 1000] * len(self.pages)

        T0 = 0.
        if len(self.pages) == 2:
            rights = [-(line.right() - O[0]) for line in pages[0]]
            lefts = [-(line.left() - O[0]) for line in pages[1]]
            T0 = (np.median(rights) + np.median(lefts)) / 2

        return np.concatenate([theta_0, a_m_0, align_0, [T0], l_m_0])

    def run(self):
        final_norm, opt_result = self.optimize()
        return self.correct(opt_result)

    def run_retry(self, n_tries=6):
        best_result = None
        best_norm = np.inf
        for _ in range(n_tries):
            final_norm, opt_result = self.optimize()
            if final_norm < best_norm:
                best_norm = final_norm
                best_result = opt_result

            if final_norm < 100:
                break
            else:
                print("**** BAD RUN. ****")

        return self.correct(best_result)

    def debug_images(self, R, g, align, l_m):
        if not lib.debug: return

        debug = cv2.cvtColor(self.im, cv2.COLOR_GRAY2BGR)
        ts_surface = E_str_project(R, g, self.base_points, 0)

        # debug_jac(theta, R, g, l_m, base_points, ts_surface)

        for Y, (_, points_XYZ) in zip(l_m, ts_surface):
            Xs, Ys, _ = points_XYZ
            # print('Y diffs:', Ys - Y)
            X_min, X_max = Xs.min(), Xs.max()
            line_Xs = np.linspace(X_min, X_max, 100)
            line_Ys = np.full((100,), Y)
            line_Zs = g(line_Xs)
            line_XYZ = np.stack([line_Xs, line_Ys, line_Zs])
            line_2d = gcs_to_image(line_XYZ, self.O, R).T
            for p0, p1 in zip(line_2d, line_2d[1:]):
                draw_line(debug, p0, p1, GREEN, 1)

        if isinstance(g, SplitPoly):
            line_Xs = np.array([g.T, g.T])
            line_Ys = np.array([-10000, 10000])
            line_Zs = g(line_Xs)
            line_XYZ = np.stack([line_Xs, line_Ys, line_Zs])
            line_2d = gcs_to_image(line_XYZ, self.O, R).T
            for p0, p1 in zip(line_2d, line_2d[1:]):
                draw_line(debug, p0, p1, RED, 4)

        for x in align.flatten():
            line_Xs = np.array([x, x])
            line_Ys = np.array([-10000, 10000])
            line_Zs = g(line_Xs)
            line_XYZ = np.stack([line_Xs, line_Ys, line_Zs])
            line_2d = gcs_to_image(line_XYZ, self.O, R).T
            draw_line(debug, line_2d[0], line_2d[1], BLUE, 4)

        lib.debug_imwrite('surface_lines.png', debug)

    def optimize(self):
        global E_str_t0s, E_align_t0s
        E_str_t0s, E_align_t0s = [], []

        n_pages = len(self.pages)
        args_0 = self.initial_args()

        x_scale = np.concatenate([
            [0.3] * 3,
            np.tile(1000 * ((3e-4 / OMEGA) ** np.arange(DEGREE)), n_pages),
            [1000, 1000] * n_pages,
            [1000],
            [1000] * len(self.base_points),
        ])

        loss = DebugLoss(
            Preproject(E_str(self.base_points, n_pages, scale_t=False)
                        + Regularize_T(self.base_points, n_pages) * 2.0,  # This just makes sure nothing crazy happens.
                        self.base_points, n_pages) \
            + make_E_align(self.pages, self.AH, self.O) * 0.2
        )

        # result = lm(
        result = opt.least_squares(
            fun=loss.residuals,
            x0=args_0,
            jac=loss.jac,
            # method='lm',
            ftol=1e-3,
            # max_nfev=1,
            # x_scale='jac',
            x_scale=x_scale,
        )

        theta, a_ms, align, T, l_m, g = unpack_args(result.x, n_pages)
        final_norm = norm(result.fun)

        print('*** OPTIMIZATION DONE ***')
        print('final norm:', final_norm)
        print('theta:', theta)
        if lib.debug:
            for a_m in a_ms:
                print('a_m:', np.concatenate([[0], a_m]))
            if isinstance(g, SplitPoly):
                print('T:', g.T)
            # print('l_m:', l_m)

        return final_norm, result

    def correct(self, opt_result):
        theta, a_ms, align, T, l_m, g = unpack_args(opt_result.x, len(self.pages))

        R = R_theta(theta)

        self.debug_images(R, g, align, l_m)

        mesh_2ds = make_mesh_2d(self.lines, self.O, R, g, n_points_w=self.n_points_w)
        result = []
        for mesh_2d in mesh_2ds:
            lib.debug_imwrite('orig.png', self.orig)
            first_pass = correct_geometry(self.orig, mesh_2d, interpolation=cv2.INTER_LANCZOS4)
            result.append(first_pass)

        return result

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    lib.debug = True
    lib.debug_prefix = ['dewarp']
    np.set_printoptions(linewidth=130, precision=4)
    out = kim2014(im)
    for i, outimg in enumerate(out):
        # gray = binarize.grayscale(outimg).astype(np.float64)
        # gray -= np.percentile(gray, 2)
        # gray *= 255 / np.percentile(gray, 95)
        # norm = binarize.ng2014_normalize(lib.clip_u8(gray))
        cv2.imwrite('dewarped{}.png'.format(i), outimg)

if __name__ == '__main__':
    go(sys.argv)
