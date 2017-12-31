import cv2
import itertools
import numpy as np

from geometry import Line

class Letter(object):
    def __init__(self, c, x, y, w, h):
        self.c = c
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def __iter__(self):
        return (x for x in self.tuple())

    def tuple(self):
        return (self.c, self.x, self.y, self.w, self.h)

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

    def box(self, im, thickness=2, color=(0, 0, 255)):
        cv2.rectangle(im, (self.x, self.y), (self.x + self.w, self.y + self.h),
                      color=color, thickness=thickness)

    def __str__(self):
        return 'Letter[{}, {}, {}, {}]'.format(self.x, self.y, self.w, self.h)

    def __repr__(self): return str(self)

class TextLine(object):
    def __init__(self, letters, model=None):
        self.letters = sorted(letters, key=lambda l: l.x)
        self.model = model

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
        self.model = None

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
        return Line.from_points(self[0].base_point(), self[-1].base_point())
