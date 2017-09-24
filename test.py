import cv2
import sys
import numpy as np
# import IPython
import matplotlib.pyplot as plt
from skimage.filters import threshold_sauvola

inpath = sys.argv[1]
original = cv2.imread(inpath, cv2.CV_LOAD_IMAGE_GRAYSCALE)

def otsu(im):
    ret, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return thresh

def gradient(im):
    space_width = len(im) / 100 * 2 + 1
    print "gradient: close width", space_width
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (space_width, 1))
    im = cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, horiz)

def vert_close(im):
    space_width = len(im) / 200 * 2 + 1
    print "vert close width", space_width
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, space_width))
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, vert)

def outline(im):
    im_w, im_h = len(im[0]), len(im)
    min_feature_size = im_h / 300
    copy = im.copy()
    cv2.rectangle(copy, (0, 0), (im_w, im_h), 255, 3)
    contours, [hierarchy] = \
        cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest hole
    max_hole = -1
    max_hole_area = 0
    i = 0
    while i >= 0:
        j = hierarchy[i][2]
        while j >= 0:
            c = contours[j]
            x, y, w, h = cv2.boundingRect(c)
            if w * h > max_hole_area:
                max_hole_area = w * h
                max_hole = j
            j = hierarchy[j][0]
        i = hierarchy[i][0]

    x, y, w, h = cv2.boundingRect(contours[max_hole])
    print "max hole:", x, y, w, h

    result = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    i = hierarchy[max_hole][2]
    while i >= 0:
        c = contours[i]
        x, y, w, h = cv2.boundingRect(c)
        # orig_slice = thresh[y:y+h, x:x+w]
        # h_projection = (orig_slice.sum(axis=0) / im_h).astype(np.uint8)
        if len(c) > 10 \
                and h < 3 * w \
                and w > min_feature_size \
                and h > min_feature_size \
                and x > 0.02 * im_w \
                and x + w < 0.98 * im_w \
                and y > 0.02 * im_h \
                and y + h < 0.98 * im_h:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 4)
        else:
            cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 4)
        i = hierarchy[i][0]
    return result

def crop(im):
    im_w, im_h = len(im[0]), len(im)
    min_feature_size = im_h / 300
    copy = im.copy()
    contours, [hierarchy] = \
        cv2.findContours(copy, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    i = 0
    crop_x0, crop_y0, crop_x1, crop_y1 = im_w, im_h, 0, 0
    result = im.copy()
    while i >= 0:
        c = contours[i]
        x, y, w, h = cv2.boundingRect(c)
        if len(c) > 10 \
                and h < 3 * w \
                and w > min_feature_size \
                and h > min_feature_size \
                and x > 0.02 * im_w \
                and x + w < 0.98 * im_w \
                and y > 0.02 * im_h \
                and y + h < 0.98 * im_h:
            crop_x0 = min(x, crop_x0)
            crop_y0 = min(y, crop_y0)
            crop_x1 = max(x + w, crop_x1)
            crop_y1 = max(y + h, crop_y1)
        i = hierarchy[i][0]

    result = cv2.bitwise_not(result)
    cv2.rectangle(result, (crop_x0, crop_y0), (crop_x1, crop_y1), 127, 4)
    crop_x0 = int(max(0, crop_x0 - .01 * im_h))
    crop_y0 = int(max(0, crop_y0 - .01 * im_h))
    crop_x1 = int(min(im_w, crop_x1 + .01 * im_h))
    crop_y1 = int(min(im_h, crop_y1 + .01 * im_h))
    return result[crop_y0:crop_y1, crop_x0:crop_x1]

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
cross55 = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

def sauvola(im):
    im = im.copy()
    global thresh
    thresh = threshold_sauvola(im, window_size=len(im) / 200)
    booleans = im > (thresh * 1.0)
    ints = booleans.astype(np.uint8) * 255
    return ints

transforms = [
    # ('Bilateral', lambda im: cv2.bilateralFilter(im, 5, 75, 41)),
    ('Gaussian', lambda im: cv2.GaussianBlur(im, (5, 5), 0)),
    ('Scale', lambda im: cv2.resize(im, (0, 0), None, 1.5, 1.5)),
    ('Clahe', lambda im: clahe.apply(im)),
    ('Threshold', sauvola),
    # ('Threshold', otsu),
    # ('Sobel', lambda im: cv2.Sobel(im, -1, 1, 1, ksize=7)),
    # ('Morph', lambda im: cv2.morphologyEx(im, cv2.MORPH_CLOSE, cross33)),
    ('Gradient', gradient),
    # ('Open', morph_open),
    ('Vertical Close', vert_close),
    ('Outline', outline),
    # ('Crop', crop),
]

transforms2 = [
    # ('Scale', lambda im: cv2.resize(im, (0, 0), None, 0.25, 0.25, cv2)),
    # ('Open', morph_open),
    ('Gradient', gradient),
    ('Outline', outline),
    # ('Crop', crop),
]

def zoom(im, frac):
    height = len(im)
    width = len(im[0])
    xlow = int(width * (0.4 - frac / 2))
    xhigh = int(width * (0.4 + frac / 2))
    ylow = int(height * (0.1 - frac / 2))
    yhigh = int(height * (0.1 + frac / 2))
    return im[ylow:yhigh, xlow:xhigh]

images = [('Original', original)]

for title, transform in transforms:
    print 'Applying', title
    images.append((title, transform(images[-1][1])))

cv2.imwrite('out2.png', images[-1][1])

# for i, (title, im) in enumerate(images):
#     plt.subplot(2, (len(images) + 1) / 2, i + 1)
#     # plt.imshow(zoom(im, 0.15), 'gray')
#     if im.dtype == np.uint8:
#         plt.imshow(im, 'gray')
#     else:
#         plt.imshow(im)
#     plt.title(title)
#     plt.xticks([]), plt.yticks([])

plt.show()
