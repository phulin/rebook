import cv2
import numpy as np

from numpy.polynomial.polynomial import Polynomial as P

BLUE = (255, 0, 0)

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

    def closest_point(self, point):
        return np.array(self.altitude(point).intersect(self))

    def distance_point(self, point):
        return np.linalg.norm(self.closest_point(point) - np.array(point))

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
        return P([self.b, self.m])

    def offset(self, offset):
        return Line.from_point_slope(self.base() + offset, self.m)

    def closest_poly_intersect(self, poly, p):
        roots = (poly - self.polynomial()).roots()
        good_roots = roots[abs(roots.imag) < 1e-10].astype(np.float64)
        good_roots_points = np.vstack([good_roots, poly(good_roots)]).T
        return good_roots_points[abs(good_roots_points - p).sum(axis=1).argmin()]

    def approx_line_poly_intersect(self, poly, approx):
        return self.closest_poly_intersect(poly, self.intersect(approx))

    def text_line_intersect(self, text_line):
        approx = Line.from_points(text_line[0].base_point(), text_line[-1].base_point())
        return self.approx_line_poly_intersect(text_line.model, approx)

    @staticmethod
    def best_intersection(lines):
        bases = [l.base() for l in lines]
        vecs = [l.vec() for l in lines]
        K = len(lines)
        I = np.eye(2)
        R = K * I - sum((np.outer(v, v.T) for v in vecs))
        q = sum(bases) - sum((v * v.dot(a)) for v, a in zip(vecs, bases))
        R_inv = np.linalg.pinv(R)
        p = np.dot(R_inv, q)
        print p
        return p

    def __str__(self):
        return 'Line[y = {:.2f}x + {:.2f}]'.format(self.m, self.b)

    def __repr__(self):
        return 'Line({}, {})'.format(self.m, self.b)

class Crop(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def nonempty(self):
        return self.x0 <= self.x1 and self.y0 <= self.y1

    def intersect(self, other):
        return Crop(
            max(self.x0, other.x0),
            max(self.y0, other.y0),
            min(self.x1, other.x1),
            min(self.y1, other.y1),
        )

    @staticmethod
    def intersect_all(crops):
        return reduce(Crop.intersect, crops)

    def union(self, other):
        return Crop(
            min(self.x0, other.x0),
            min(self.y0, other.y0),
            max(self.x1, other.x1),
            max(self.y1, other.y1),
        )

    @staticmethod
    def union_all(crops):
        return reduce(Crop.union, crops)

    def apply(self, im):
        assert self.nonempty()
        return im[self.y0:self.y1, self.x0:self.x1]

    @staticmethod
    def full(im):
        h, w = im.shape
        return Crop(0, 0, w, h)

    @staticmethod
    def null(im):
        h, w = im.shape
        return Crop(w, h, 0, 0)

    @staticmethod
    def from_rect(x, y, w, h):
        return Crop(x, y, x + w, y + h)

    def draw(self, im, color=BLUE, thickness=2):
        cv2.rectangle(im, (int(self.x0), int(self.y0)), (int(self.x1), int(self.y1)),
                      color=color, thickness=thickness)

    def __repr__(self):
        return "Crop({}, {}, {}, {})".format(self.x0, self.y0, self.x1, self.y1)
