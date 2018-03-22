import cv2
import numpy as np

from dewarp import get_AH_lines, correct_geometry, estimate_vanishing, \
    arc_length_points
from geometry import Line
import lib
from lib import RED, GREEN, BLUE

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
