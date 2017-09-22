import cv2
import sys
import numpy as np
import IPython
import matplotlib.pyplot as plt
from skimage.filters import (threshold_otsu, threshold_niblack,
                             threshold_sauvola)

inpath = sys.argv[1]
original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def do_thresh(im):
    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def outline(im):
    copy = im.copy()
    contours, hierarchy = cv2.findContours(copy, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    result = im.copy()
    for c in contours:
        if len(c) <= 10: continue
        x, y, w, h = cv2.boundingRect(c)
        # ratio = float(w) / float(h)
        # if ratio > 10 or ratio < 0.1: continue
        cv2.rectangle(result, (x, y), (x + w, y + h), 127, 3)
    return result

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(100, 20))
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

def sauvola(im):
    thresh = threshold_sauvola(im, window_size=15)
    booleans = im > (thresh * 1.1)
    ints = booleans.astype(np.uint8) * 255
    return ints

transforms = [
    # ('Gaussian', lambda im: cv2.GaussianBlur(im, (5, 5), 0)),
    # ('Bilateral', lambda im: cv2.bilateralFilter(im, 5, 75, 41)),
    ('Scale', lambda im: cv2.resize(im, (0, 0), None, 1.5, 1.5)),
    ('Clahe', lambda im: clahe.apply(im)),
    ('Threshold', sauvola),
    # ('Threshold', do_thresh),
    # ('Sobel', lambda im: cv2.Sobel(im, -1, 1, 1, ksize=7)),
    # ('Morph', lambda im: cv2.morphologyEx(im, cv2.MORPH_CLOSE, element)),
    # ('Outline', outline),
]

def zoom(im, frac):
    height = len(im)
    width = len(im[0])
    xlow = int(width * (0.75 - frac / 2))
    xhigh = int(width * (0.75 + frac / 2))
    ylow = int(height * (0.5 - frac / 2))
    yhigh = int(height * (0.5 + frac / 2))
    return im[ylow:yhigh, xlow:xhigh]

images = [('Original', original)]

for title, transform in transforms:
    print 'Applying', title
    images.append((title, transform(images[-1][1])))

for i, (title, im) in enumerate(images):
    plt.subplot(2, (len(images) + 1) / 2, i + 1)
    plt.imshow(zoom(im, 0.15), 'gray')
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.show()
