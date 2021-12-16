from __future__ import division, print_function

import cv2
import numpy as np

from numpy.polynomial.polynomial import Polynomial as P
from functools import reduce

BLUE = (255, 0, 0)

def closest_root_to(poly_on, poly_root, p):
    roots = poly_root.roots()
    good_roots = roots[abs(roots.imag) < 1e-10].real.astype(np.float64)
    good_roots_points = np.vstack([good_roots, poly_on(good_roots)]).T
    return good_roots_points[abs(good_roots_points - p).sum(axis=1).argmin()]

class Line(object):
    def __init__(self, m, b):
        self.m = m
        self.b = b

    def __call__(self, x):
        return self.m * x + self.b

    @staticmethod
    def from_polynomial(p):
        b, m = p.coef
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

    @staticmethod
    def homogeneous(A, B, C):
        # Ax + By = C
        return Line(-A / B, C / B)

    @staticmethod
    def fit(points):
        if type(points) == list: points = np.array(points)

        poly = P.fit(points[:, 0], points[:, 1], 1, domain=[-1, 1])
        return Line.from_polynomial(poly)

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

    # Returns angle in range [-pi/2, pi/2)
    def angle(self):
        return np.arctan(self.m)

    def polynomial(self):
        return P([self.b, self.m])

    def offset(self, offset):
        return Line.from_point_slope(self.base() + offset, self.m)

    def closest_poly_intersect(self, poly, p):
        return closest_root_to(poly, poly - self.polynomial(), p)

    def approx_line_poly_intersect(self, poly, approx):
        return self.closest_poly_intersect(poly, self.intersect(approx))

    def text_line_intersect(self, text_line):
        return self.approx_line_poly_intersect(text_line.model, text_line.approx_line())

    @staticmethod
    def best_intersection(lines):
        bases = [l.base() for l in lines]
        vecs = [l.vector() for l in lines]
        K = len(lines)
        I = np.eye(2)
        R = K * I - sum((np.outer(v, v.T) for v in vecs))
        q = sum(bases) - sum((v * v.dot(a)) for v, a in zip(vecs, bases))
        R_inv = np.linalg.pinv(R)
        p = np.dot(R_inv, q)
        return p

    def __str__(self):
        return 'Line[y = {:.2f}x + {:.2f}]'.format(self.m, self.b)

    def __repr__(self):
        return 'Line({}, {})'.format(self.m, self.b)

class Line3D(object):
    # p0, p1: points on line
    def __init__(self, p0, p1):
        self.p0, self.p1 = p0, p1

    @staticmethod
    def from_coords(x0, y0, z0, x1, y1, z1):
        return Line3D(np.array((x0, y0, z0)), np.array((x1, y1, z1)))

    @staticmethod
    def from_point_vec(p, v):
        p = np.atleast_1d(p)
        v = np.atleast_1d(v)
        return Line3D(p, p + v)

    @property
    def vec(self):
        return self.p1 - self.p0

    def transform(self, A):
        return Line3D(A.dot(self.p0), A.dot(self.p1))

    def offset(self, v):
        return Line3D(self.p0 + v, self.p1 + v)

    def project(self, z):
        return Line.from_points(self.p0[0:2] * (z / self.p0[2]),
                                self.p1[0:2] * (z / self.p1[2]))

    def __str__(self):
        return 'Line[({:.2f}, {:.2f}, {:.2f}) => ({:.2f}, {:.2f}, {:.2f})]'.format(
            self.p0[0], self.p0[1], self.p0[2],
            self.p1[0], self.p1[1], self.p1[2]
        )

class Crop(object):
    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = x0, y0
        self.x1, self.y1 = x1, y1

    @staticmethod
    def from_points(points):
        return Crop(points[0].min(), points[1].min(),
                    points[0].max(), points[1].max())

    @staticmethod
    def from_line(line):
        result = Crop(
            min([letter.left() for letter in line]),
            min([letter.top() for letter in line]),
            max([letter.right() for letter in line]),
            max([letter.bottom() for letter in line])
        )
        if line.underlines:
            result = result.union(Crop.union_all((under.crop() for under in line.underlines)))

        return result

    @staticmethod
    def from_lines(lines):
        return Crop.union_all((Crop.from_line(line) for line in lines))

    @property
    def w(self):
        return self.x1 - self.x0

    @property
    def h(self):
        return self.y1 - self.y0

    def corners(self):
        return [
            (self.x0, self.y0),
            (self.x0, self.y1),
            (self.x1, self.y0),
            (self.x1, self.y1),
        ]

    def nonempty(self):
        return self.x0 < self.x1 and self.y0 < self.y1

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
        crop = self.intersect(Crop.full(im))
        return im[crop.y0:crop.y1, crop.x0:crop.x1]

    @staticmethod
    def full(im):
        h, w = im.shape[:2]
        return Crop(0, 0, w, h)

    @staticmethod
    def null(im):
        h, w = im.shape[:2]
        return Crop(w, h, 0, 0)

    @staticmethod
    def from_rect(x, y, w, h):
        return Crop(x, y, x + w, y + h)

    @staticmethod
    def from_whitespace(im):
        assert im.min() < im.max()
        immax = im == im.max()

        def first_nonmax_row(a):
            return a.min(axis=1).argmin()

        x0 = first_nonmax_row(immax.T)
        y0 = first_nonmax_row(immax)
        x1 = immax.shape[1] - first_nonmax_row(immax.T[::-1])
        y1 = immax.shape[0] - first_nonmax_row(immax[::-1])

        return Crop(x0, y0, x1, y1)

    @staticmethod
    def remove_whitespace(im):
        return Crop.from_whitespace(im).apply(im)

    def expand(self, factor):
        return Crop(
            int(round(self.x0 - self.w * factor)),
            int(round(self.y0 - self.h * factor)),
            int(round(self.x1 + self.w * factor)),
            int(round(self.y1 + self.h * factor))
        )

    def draw(self, im, color=BLUE, thickness=2):
        cv2.rectangle(im, (int(self.x0), int(self.y0)), (int(self.x1), int(self.y1)),
                      color=color, thickness=thickness)

    def __repr__(self):
        return "Crop({}, {}, {}, {})".format(self.x0, self.y0, self.x1, self.y1)

    def __iter__(self):
        return iter((self.x0, self.y0, self.x1, self.y1))
