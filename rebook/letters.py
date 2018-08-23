from __future__ import division, print_function

import cv2
import itertools
import numpy as np
from numpy.polynomial import Polynomial as Poly
from skimage.measure import ransac

from geometry import Crop, Line

class Letter(object):
    def __init__(self, label, label_map, stats, centroid):
        self.label = label
        self.label_map = label_map
        self.stats = stats
        self.centroid = centroid

    @property
    def x(self): return self.stats[cv2.CC_STAT_LEFT]

    @property
    def y(self): return self.stats[cv2.CC_STAT_TOP]

    @property
    def w(self): return self.stats[cv2.CC_STAT_WIDTH]

    @property
    def h(self): return self.stats[cv2.CC_STAT_HEIGHT]

    def area(self):
        return self.stats[cv2.CC_STAT_AREA]

    def __iter__(self):
        return (x for x in self.tuple())

    def tuple(self):
        return (self.x, self.y, self.w, self.h)

    def left(self):
        return self.x

    def right(self):
        return self.x + self.w

    def top(self):
        return self.y

    def bottom(self):
        return self.y + self.h

    def left_mid(self):
        return np.array((self.x, self.y + self.h / 2.0))

    def right_mid(self):
        return np.array((self.x + self.w, self.y + self.h / 2.0))

    def left_bot(self):
        return np.array((self.x, self.y + self.h))

    def right_bot(self):
        return np.array((self.x + self.w, self.y + self.h))

    def corners(self):
        return np.array((
            (self.x, self.y),
            (self.x, self.y + self.h),
            (self.x + self.w, self.y),
            (self.x + self.w, self.y + self.h)
        ))

    def base_point(self):
        return np.array((self.x + self.w / 2.0, self.y + self.h))

    def top_point(self):
        return np.array((self.x + self.w / 2.0, self.y))

    def crop(self):
        return Crop(self.x, self.y, self.x + self.w, self.y + self.h)

    def slice(self, im):
        return self.crop().apply(im)

    def raster(self):
        sliced = self.crop().apply(self.label_map)
        return sliced == self.label

    def top_contour(self):
        return self.y + self.raster().argmax(axis=0)

    def bottom_contour(self):
        return self.y + self.h - 1 - self.raster()[::-1].argmax(axis=0)

    def box(self, im, color=(0, 0, 255), thickness=2):
        cv2.rectangle(im, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      color=color, thickness=thickness)

    def __str__(self):
        return 'Letter[{}, {}, {}, {}]'.format(self.x, self.y, self.w, self.h)

    def __repr__(self): return str(self)

class TextLine(object):
    def __init__(self, letters, model=None, underlines=None):
        self.letters = sorted(letters, key=lambda l: l.x)
        self.model = model
        self.model_line = None
        self._inliers = None
        self._line_inliers = None
        self.underlines = underlines if underlines is not None else []

    def __iter__(self):
        return (l for l in self.letters)

    def __len__(self):
        return len(self.letters)

    def __getitem__(self, key):
        return self.letters[key]

    def __str__(self):
        return str(self.letters)

    def __call__(self, value):
        assert self.model is not None
        return self.model(value)

    def __add__(self, other):
        return self.letters + other.letters

    def copy(self):
        return TextLine(self.letters[:], self.model)

    def compress(self, flags):
        assert len(flags) == len(self.letters)
        self.letters = list(itertools.compress(self.letters, flags))

    def merge(self, other):
        self.letters += other.letters
        self.letters.sort(key=lambda l: l.x)
        self.underlines = list(set(self.underlines) | set(other.underlines))
        self.model = None
        self._inliers = None
        self.model_line = None
        self._line_inliers = None

    def domain(self):
        return self.letters[0].base_point()[0], self.letters[-1].base_point()[0]

    def left(self):
        return self.letters[0].left()

    def right(self):
        return self.letters[-1].right()

    def width(self):
        return self.right() - self.left()

    def first_base(self):
        return self.letters[0].base_point()

    def last_base(self):
        return self.letters[-1].base_point()

    def approx_line(self):
        return Line.from_points(self.first_base(), self.last_base())

    def base_points(self):
        return np.array([l.base_point() for l in self.letters])

    def crop(self):
        if self.underlines:
            return Crop.union(
                Crop.union_all([l.crop() for l in self.letters]),
                Crop.union_all([u.crop() for u in self.underlines]),
            )
        else:
            return Crop.union_all([l.crop() for l in self.letters])

    class PolyModel5(object):
        def estimate(self, data):
            self.params = Poly.fit(data[:, 0], data[:, 1], 5, domain=[-1, 1])
            return True

        def residuals(self, data):
            return abs(self.params(data[:, 0]) - data[:, 1])

    def fit_poly(self):
        if self.model is None:
            model, inliers = ransac(self.base_points(), TextLine.PolyModel5, 10, 4)
            self.model = model.params
            self._inliers = list(itertools.compress(self.letters, inliers))

        return self.model

    class LineModel(object):
        def estimate(self, data):
            self.params = Line.fit(data)
            return True

        def residuals(self, data):
            return np.abs(self.params(data[:, 0]) - data[:, 1])

    def fit_line(self):
        if self.model_line is None:
            if len(self) <= 3:
                self.model_line = Line.fit(self.base_points())
            else:
                model, inliers = ransac(self.base_points(), TextLine.LineModel, 3, 4)
                self.model_line = model.params
                self._line_inliers = list(itertools.compress(self.letters, inliers))

        return self.model_line

    def inliers(self):
        self.fit_poly()
        return self._inliers

    def line_inliers(self):
        self.fit_line()
        return self._line_inliers

class Underline(object):
    def __init__(self, label, label_map, stats):
        self.label = label
        self.label_map = label_map
        self.stats = stats

    @property
    def x(self): return self.stats[cv2.CC_STAT_LEFT]

    @property
    def y(self): return self.stats[cv2.CC_STAT_TOP]

    @property
    def w(self): return self.stats[cv2.CC_STAT_WIDTH]

    @property
    def h(self): return self.stats[cv2.CC_STAT_HEIGHT]

    def crop(self):
        return Crop(self.x, self.y, self.x + self.w, self.y + self.h)
