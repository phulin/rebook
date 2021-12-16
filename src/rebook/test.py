import cv2
import sys
# import IPython
import numpy as np
import matplotlib.pyplot as plt

from . import binarize
from .algorithm import text_contours, dewarp

inpath = sys.argv[1]
original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def outline(im):
    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)

    good_contours, bad_contours = text_contours(im)

    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 4)
    for c in bad_contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 4)

    return result

def crop(im):
    im_w, im_h = len(im[0]), len(im)

    good_contours, _ = text_contours(im)
    crop_x0, crop_y0, crop_x1, crop_y1 = im_w, im_h, 0, 0
    for c in good_contours:
        x, y, w, h = cv2.boundingRect(c)
        crop_x0 = min(x, crop_x0)
        crop_y0 = min(y, crop_y0)
        crop_x1 = max(x + w, crop_x1)
        crop_y1 = max(y + h, crop_y1)

    result = cv2.bitwise_not(im)
    cv2.rectangle(result, (crop_x0, crop_y0), (crop_x1, crop_y1), 127, 4)
    crop_x0 = int(max(0, crop_x0 - .01 * im_h))
    crop_y0 = int(max(0, crop_y0 - .01 * im_h))
    crop_x1 = int(min(im_w, crop_x1 + .01 * im_h))
    crop_y1 = int(min(im_h, crop_y1 + .01 * im_h))
    return result[crop_y0:crop_y1, crop_x0:crop_x1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))

transforms = [
    # ('Bilateral', lambda im: cv2.bilateralFilter(im, 5, 75, 41)),
    # ('Gaussian', lambda im: cv2.GaussianBlur(im, (9, 9), 0)),
    # ('Scale', lambda im: cv2.resize(im, (0, 0), None, 4.0, 4.0)),
    # ('Clahe', lambda im: clahe.apply(im)),
    # ('Sobel', lambda im: cv2.Sobel(im, -1, 1, 1, ksize=7)),
    # ('Morph', lambda im: cv2.morphologyEx(im, cv2.MORPH_CLOSE, cross33)),
    # ('Gradient', gradient),
    # ('Open', morph_open),
    # ('Vertical Close', vert_close),
    # ('Outline', outline),
    # ('Crop', crop),
    ('AOtsu', binarize.adaptive_otsu),
    ('Dewarp', dewarp),
]

options = [
    # ('Sauvola-1.0/Clahe', lambda im: sauvola(clahe.apply(im), thresh_factor=1.0)),
    # ('Sauvola-1.0', sauvola),
    # ('Ntiro2014', binarize.ntirogiannis2014),
    # ('Roth', binarize.roth),
    # ('Kamel/Zhao', kamel),
    # ('Lu2010', lu2010),
    # ('Yan-0.3', lambda im: binarize.yan(im, alpha=0.3)),
    # ('Yan-0.65', lambda im: yan(im, alpha=0.65)),
]

def zoom(im, frac):
    height = len(im)
    width = len(im[0])
    xlow = int(width * (0.7 - frac / 2))
    xhigh = int(width * (0.7 + frac / 2))
    ylow = int(height * (0.3 - frac / 2))
    yhigh = int(height * (0.3 + frac / 2))
    return im[ylow:yhigh, xlow:xhigh]

transformed = [('Original', original)]
images = [('Original', original)]

for title, transform in transforms:
    print('Applying', title)
    transformed.append((title, transform(images[-1][1])))

# images = transformed

_, last_image = images[-1]
for title, option in options:
    print('Applying', title)
    images.append((title, option(last_image)))

cv2.imwrite('out2.png', images[-1][1])

for i, (title, im) in enumerate(images):
    plt.subplot(2, (len(images) + 1) / 2, i + 1)
    im = zoom(im, 0.1)
    if im.dtype == np.uint8:
        plt.imshow(im, 'gray')
    else:
        plt.imshow(im)
    plt.title(title)
    plt.xticks([]), plt.yticks([])

plt.show()
