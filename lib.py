import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
# import IPython
import math
from skimage.filters import threshold_sauvola, threshold_niblack

debug = True
def debug_imwrite(path, im):
    if debug:
        cv2.imwrite(path, im)

def normalize_u8(im):
    im_max = im.max()
    alpha = 255 / im_max
    beta = im.min() * im_max / 255
    return cv2.convertScaleAbs(im, alpha=alpha, beta=beta)

def clip_u8(im):
    return im.clip(0, 255).astype(np.uint8)

def bool_to_u8(im):
    return im.astype(np.uint8) - 1

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
def gradient(im):
    return cv2.morphologyEx(im, cv2.MORPH_GRADIENT, cross33)

def vert_close(im):
    space_width = len(im) / 200 * 2 + 1
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, space_width))
    return cv2.morphologyEx(im, cv2.MORPH_CLOSE, vert)

def sauvola(im, window_factor=200, k=0.2, thresh_factor=1.0):
    thresh = threshold_sauvola(im, window_size=len(im) / window_factor * 2 + 1)
    booleans = im > (thresh * thresh_factor)
    ints = booleans.astype(np.uint8) * 255
    return ints

def niblack(im):
    thresh = threshold_niblack(im, window_size=len(im) / 200 * 2 + 1)
    booleans = im > (thresh * 1.0)
    ints = booleans.astype(np.uint8) * 255
    return ints

def kittler(im):
    h, g = np.histogram(im.ravel(), 256, [0, 256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g**2)
    sigma_f = np.sqrt(s/c - (m/c)**2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb/cb - (mb/cb)**2)
    p = c / c[-1]
    v = p * np.log(sigma_f) + (1-p)*np.log(sigma_b) - \
        p*np.log(p) - (1-p)*np.log(1-p)
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    _, thresh = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)
    return thresh

def roth(im, s=51, t=0.8):
    im_h, im_w = im.shape
    means = cv2.blur(im, (s, s))
    booleans = im > means * t
    ints = booleans.astype(np.uint8) * 255
    debug_imwrite('roth.png', ints)
    return ints

# s = stroke width
def kamel(im, s=None, T=25):
    im_h, im_w = im.shape
    if s is None or s <= 0:
        s = im_h / 200
    size = 2 * s + 1
    means = cv2.blur(im, (size, size), borderType=cv2.BORDER_REFLECT)
    padded = np.pad(means, (s, s), 'edge')
    im_plus_T = im.astype(np.int64) + T
    im_plus_T = im_plus_T.clip(min=0, max=255).astype(np.uint8)
    L1 = padded[0:im_h, 0:im_w] <= im_plus_T
    L2 = padded[0:im_h, s:im_w + s] <= im_plus_T
    L3 = padded[0:im_h, 2 * s:im_w + 2 * s] <= im_plus_T
    L4 = padded[s:im_h + s, 2 * s:im_w + 2 * s] <= im_plus_T
    L5 = padded[2 * s:im_h + 2 * s, 2 * s:im_w + 2 * s] <= im_plus_T
    L6 = padded[2 * s:im_h + 2 * s, s:im_w + s] <= im_plus_T
    L7 = padded[2 * s:im_h + 2 * s, 0:im_w] <= im_plus_T
    L0 = padded[s:im_h + s, 0:im_w] <= im_plus_T
    L04, L15, L26, L37 = L0 & L4, L1 & L5, L2 & L6, L3 & L7
    b = (L04 & L15) | (L15 & L26) | (L26 & L37) | (L37 & L04)

    return b.astype(np.uint8) * 255

def row_zero_run_lengths(row):
    bounded = np.hstack(([255], row, [255]))
    diffs = np.diff(bounded)
    run_starts, = np.where(diffs < 0)
    run_ends, = np.where(diffs > 0)
    return run_ends - run_starts

def horiz_zero_run_lengths(im):
    return np.hstack(map(row_zero_run_lengths, im))

def yan(im, alpha=0.4):
    im_h, im_w = im.shape
    first_pass = otsu(im)

    horiz_runs = horiz_zero_run_lengths(first_pass)
    vert_runs = horiz_zero_run_lengths(first_pass.T)
    run_length_hist, _ = np.histogram(np.hstack((horiz_runs, vert_runs)),
                                      bins=np.arange(0, im_h / 100))
    argmax = run_length_hist.argmax()
    candidates, = np.where(run_length_hist[argmax:argmax+3] >
                           run_length_hist[argmax] * .8)
    stroke_width = candidates.max() + argmax
    print 'stroke width:', stroke_width

    size = 2 * stroke_width + 1
    means = cv2.blur(im, (size, size),
                     borderType=cv2.BORDER_REFLECT)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    maxes = cv2.morphologyEx(im, cv2.MORPH_DILATE, element).astype(float)
    mins = cv2.morphologyEx(im, cv2.MORPH_ERODE, element).astype(float)
    # if means closer to max, probably more noisy gray levels
    assert (maxes >= means).all() and (means >= mins).all()
    Ts = np.where(maxes - means > means - mins,
                  alpha / 3 * (mins + mins + means),
                  alpha / 3 * (mins + means + means))

    result = kamel(im, s=stroke_width, T=Ts)
    debug_imwrite('yan_{}.png'.format(alpha), result)
    return result

def median_downsample(row, step):
    row_w, = row.shape
    small_w = row_w / step
    row_w_even = row_w * step
    reshaped = row[:row_w_even].reshape(small_w, step)
    return np.median(reshaped, axis=1)

def median_downsample_row(im, step, percentile=50):
    im_h, im_w = im.shape
    small_w = im_w / step
    im_w_even = small_w * step
    reshaped = im[:, :im_w_even].reshape(im_h, small_w, step)
    return np.percentile(reshaped, percentile, axis=2)

def polynomial_background_row(row, order_0=6, threshold=10):
    row_w, = row.shape

    step = row_w / 500  # target this many points in interpolation of rowage
    small = median_downsample(row, step)
    small_w, = small.shape
    order_f = float(order_0)

    while True:
        fitted = poly.polyfit(np.arange(small_w), small.T, 50)
        values = poly.polyval(np.arange(0, small_w, 1.0 / step), fitted)
        difference = np.abs(fitted - values)
        if difference.max() <= threshold:  # or len(xs) < order:
            break
        order_f += 0.2
    return values

def polynomial_background_easy(im, order=10):
    _, im_w = im.shape
    step = im_w / 100
    halfstep = step / 2.0
    small = median_downsample_row(im, step)
    _, small_w = small.shape
    xs = halfstep + np.arange(0, small_w * step, step)
    fitted = poly.polyfit(xs, small.T, order)
    values = poly.polyval(np.arange(im_w), fitted)
    return values

def nonzero_distances_row(im):
    indices, = np.where(im.flatten() > 0)
    return np.diff(indices)

def lu2010(im):
    im_h, im_w = im.shape

    im_bg_row = polynomial_background_easy(im)  # TODO: implement full
    im_bg = polynomial_background_easy(im_bg_row.T).T
    im_bg = im_bg.clip(0.1, 255)
    debug_imwrite('bg.png', im_bg)

    C = np.percentile(im, 30)
    I_bar = clip_u8(C / im_bg * im)
    debug_imwrite('ibar.png', I_bar)
    I_bar_padded = np.pad(I_bar, (1, 1), 'edge')

    V_h = cv2.absdiff(I_bar_padded[2:, 1:-1], I_bar_padded[:-2, 1:-1])
    V_v = cv2.absdiff(I_bar_padded[1:-1, 2:], I_bar_padded[1:-1, :-2])
    V = V_h + V_v
    V[V < V_h] = 255  # clip overflow
    _, stroke_edges = cv2.threshold(V, 0, 255,
                                    cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 1 if high contrast, 0 otherwise.
    E_inv = stroke_edges & 1
    E_inv_255 = E_inv * 255
    debug_imwrite('e_inv.png', E_inv_255)
    im_high = im & E_inv_255
    debug_imwrite('im_high.png', im_high)

    # calculate stroke width
    H, _ = np.histogram(nonzero_distances_row(E_inv), np.arange(im_h / 100))
    H[1] = 0  # don't use adjacent pix
    W = H.argmax()
    print 'stroke width:', W
    size = 2 * W
    N_min = W

    if size >= 16:
        E_inv = E_inv.astype(np.uint16)
    window = (size | 1, size | 1)
    N_e = cv2.boxFilter(E_inv, -1, window, normalize=False)
    debug_imwrite('n_e.png', bool_to_u8(N_e >= N_min))

    E_mean = cv2.boxFilter(im_high, cv2.CV_32S, window,
                           normalize=False) / N_e
    debug_imwrite('i_bar_e_mean.png', bool_to_u8(I_bar <= E_mean))

    out = bool_to_u8((N_e >= N_min) & (I_bar <= E_mean))
    debug_imwrite('lu2010.png', out)
    return out

def otsu(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    debug_imwrite('otsu.png', thresh)
    return thresh

def hsl_gray(im):
    assert len(im.shape) == 3
    hls = cv2.cvtColor(im, cv2.COLOR_RGB2HLS)
    _, l, s = cv2.split(hls)
    return s, l

def text_contours(im):
    im_w, im_h = len(im[0]), len(im)
    min_feature_size = im_h / 300

    copy = im.copy()
    cv2.rectangle(copy, (0, 0), (im_w, im_h), 255, 3)
    contours, [hierarchy] = \
        cv2.findContours(copy, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # find biggest holes
    image_area = im_w * im_h
    good_holes = []
    i = 0
    while i >= 0:
        j = hierarchy[i][2]
        while j >= 0:
            c = contours[j]
            x, y, w, h = cv2.boundingRect(c)
            if w * h > image_area * 0.25:
                good_holes.append(j)
            j = hierarchy[j][0]
        i = hierarchy[i][0]

    good_contours, bad_contours = [], []
    for hole in good_holes:
        x, y, w, h = cv2.boundingRect(contours[hole])
        # print "hole:", x, y, w, h

        i = hierarchy[hole][2]
        while i >= 0:
            c = contours[i]
            x, y, w, h = cv2.boundingRect(c)
            orig_slice = im[y:y + h, x:x + w]
            # print 'mean:', orig_slice.mean(), 'horiz stddev:', orig_slice.mean(axis=0).std()
            # print 'contour:', x, y, w, h
            if len(c) > 10 \
                    and h < 2 * w \
                    and w > min_feature_size \
                    and h > min_feature_size:
                good_contours.append(c)
            else:
                bad_contours.append(c)
            i = hierarchy[i][0]

    return good_contours, bad_contours

# and x > 0.02 * im_w \
# and x + w < 0.98 * im_w \
# and y > 0.02 * im_h \
# and y + h < 0.98 * im_h:

def binarize(im, algorithm=otsu, resize=1.0):
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(5, 5))
    if (im + 1 < 2).all():  # black and white
        return im
    else:
        if resize < 0.99 or resize > 1.01:
            im = cv2.resize(im, (0, 0), None, resize, resize)
        if len(im.shape) > 2:
            sat, lum = hsl_gray(im)
            # sat, lum = clahe.apply(sat), clahe.apply(lum)
            return algorithm(lum)  # & yan(l, T=35)
        else:
            # img = clahe.apply(img)
            # cv2.imwrite('clahe.png', img)
            return algorithm(im)

def skew_angle(im):
    im_h, _ = im.shape

    first_pass = binarize(im, algorithm=roth, resize=1000.0 / im_h)

    grad = gradient(first_pass)
    space_width = (im_h / 50) | 1
    line_height = (im_h / 400) | 1
    horiz = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                      (space_width, line_height))
    grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, horiz)

    lines = cv2.cvtColor(grad, cv2.COLOR_GRAY2RGB)
    line_contours, _ = text_contours(grad)
    alphas = []
    for c in line_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 4 * h:
            vx, vy, x1, y1 = cv2.fitLine(c, cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
            cv2.line(lines,
                     (x1 - vx * 1000, y1 - vy * 1000),
                     (x1 + vx * 1000, y1 + vy * 1000),
                     (255, 0, 0), thickness=3)
            alphas.append(math.atan2(vy, vx))
    debug_imwrite('lines.png', lines)
    return np.median(alphas)

def safe_rotate(im, angle):
    debug_imwrite('prerotated.png', im)
    im_h, im_w = im.shape
    if abs(angle) > math.pi / 4:
        print "warning: too much rotation"
        return im

    im_h_new = im_w * abs(math.sin(angle)) + im_h * math.cos(angle)
    im_w_new = im_h * abs(math.sin(angle)) + im_w * math.cos(angle)

    pad_h = int(math.ceil((im_h_new - im_h) / 2))
    pad_w = int(math.ceil((im_w_new - im_w) / 2))

    padded = np.pad(im, (pad_h, pad_w), 'constant', constant_values=255)
    padded_h, padded_w = padded.shape
    angle_deg = angle * 180 / math.pi
    print 'rotating to angle:', angle_deg, 'deg'
    matrix = cv2.getRotationMatrix2D((padded_w / 2, padded_h / 2), angle_deg, 1)
    result = cv2.warpAffine(padded, matrix, (padded_w, padded_h),
                            borderMode=cv2.BORDER_CONSTANT,
                            borderValue=255)
    debug_imwrite('rotated.png', result)
    return result
