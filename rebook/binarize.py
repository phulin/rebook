import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import sys

from algorithm import fast_stroke_width
import inpaint
import lib
from lib import normalize_u8, clip_u8, bool_to_u8, debug_imwrite

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
rect33 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

def hls_gray(im):
    assert len(im.shape) == 3
    hls = cv2.cvtColor(im, cv2.COLOR_BGR2HLS)
    _, l, _ = cv2.split(hls)
    return l

def CIELab_gray(im):
    assert len(im.shape) == 3
    Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    L, _, _ = cv2.split(Lab)
    return L

@lib.timeit
def pca_gray(im):
    assert len(im.shape) == 3
    Lab = cv2.cvtColor(im, cv2.COLOR_BGR2Lab)
    im_1d = Lab.reshape(im.shape[0] * im.shape[1], 3).astype(np.float32)
    im_1d -= np.mean(im_1d)
    U, S, V = np.linalg.svd(im_1d, full_matrices=False)
    coeffs = V[0]
    if coeffs[0] < 0:
        coeffs = -coeffs
    result = normalize_u8(np.tensordot(Lab, coeffs, axes=1))
    lib.debug_imwrite('pca.png', result)
    return result

def otsu(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    debug_imwrite('otsu.png', thresh)
    return thresh

def teager(im):
    im32 = im.astype(np.int32)
    padded = np.pad(im32, (1, 1), 'edge')

    return normalize_u8(3 * im32 ** 2 - padded[2:, 2:] * padded[:-2, :-2] / 2 \
        - padded[2:, :-2] * padded[:-2, 2:] / 2 \
        - padded[2:, 1:-1] * padded[:-2, 1:-1] \
        - padded[1:-1, 2:] * padded[1:-1, :-2])

@lib.timeit
def ntirogiannis2014(im):
    im_h, _ = im.shape
    IM = cv2.erode(niblack(im, window_size=61, k=0.2), rect33)
    debug_imwrite('niblack.png', IM)

    inpainted, modified = inpaint.inpaint_ng14(im, -IM)
    # TODO: get inpainted with avg as well
    debug_imwrite('inpainted.png', inpainted)

    bg = (inpainted & ~IM) | (modified & IM)
    debug_imwrite('bg.png', bg)

    im_f = im.astype(float) + 1
    bg_f = bg.astype(float) + 1
    F = im_f / bg_f
    N = clip_u8(255 * (F - F.min()))
    debug_imwrite('N.png', N)
    O = otsu(im)

    # TODO: use actual skeletonization system and CC filtering
    strokes = fast_stroke_width(O)
    debug_imwrite('strokes.png', normalize_u8(strokes.clip(0, 10)))
    SW = int(round(strokes.sum() / np.count_nonzero(strokes)))
    print 'SW =', SW

    BG_count = np.count_nonzero(O)
    FG_count = O.size - BG_count
    O_16 = O.astype(np.int16)
    O_inv = ~O
    O_inv_16 = O_inv.astype(np.int16)

    FG = (O_inv & N).astype(np.int16)
    FG_avg = FG.sum() / FG_count
    FG_std = (O_inv_16 & abs(FG - FG_avg)).sum() / float(FG_count)

    BG = (O & N).astype(np.int16)
    BG_avg = BG.sum() / BG_count
    BG_std = (O_16 & abs(BG - BG_avg)).sum() / float(BG_count)

    C = -50 * np.log10((FG_avg + FG_std) / (BG_avg - BG_std))
    k = -0.2 - 0.1 * np.floor(C / 10)
    print 'sauvola:', C, k
    local = sauvola(N, window_size=2 * SW + 1, k=k)
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 1))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
    O_eroded = cv2.erode(cv2.erode(O, horiz), vert)

    # TODO: implement CC-based algorithm
    # take everything that's FG in O_eroded and niblack
    return O_eroded | local

# @lib.timeit
# def sauvola(im, window_size=61, k=0.2, thresh_factor=1.0):
#     thresh = threshold_sauvola(im, k=k, window_size=window_size)
#     booleans = im > (thresh * thresh_factor)
#     ints = booleans.astype(np.uint8) * 255
#     return ints
# 
# @lib.timeit
# def niblack(im, window_size=61, k=0.2):
#     thresh = threshold_niblack(im, window_size=window_size, k=k)
#     booleans = im > (thresh * 1.0)
#     return -booleans.astype(np.uint8)

def mean_std(im, W):
    s = W / 2
    N = W * W

    padded = np.pad(im, (s, s), 'reflect')
    sum1, sum2 = cv2.integral2(padded, sdepth=cv2.CV_32F)

    S1 = sum1[W:, W:] - sum1[W:, :-W] - sum1[:-W, W:] + sum1[:-W, :-W]
    S2 = sum2[W:, W:] - sum2[W:, :-W] - sum2[:-W, W:] + sum2[:-W, :-W]

    means = S1 / N

    variances = S2 / N - means * means
    stds = np.sqrt(variances.clip(0, None))

    return means, stds

@lib.timeit
def niblack(im, window_size=61, k=0.2):
    means, stds = mean_std(im, window_size)
    thresh = means + k * stds

    return -(im > thresh).astype(np.uint8)

@lib.timeit
def sauvola(im, window_size=61, k=0.2):
    assert im.dtype == np.uint8
    means, stds = mean_std(im, window_size)
    thresh = means * (1 + k * ((stds / 127) - 1))

    return -(im > thresh).astype(np.uint8)

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
    ints = cv2.bitwise_not(bool_to_u8(im > means * t))
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
    first_pass = adaptive_otsu(im)

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

    # im_bg_row = polynomial_background_easy(im)  # TODO: implement full
    # im_bg = polynomial_background_easy(im_bg_row.T).T
    # im_bg = im_bg.clip(0.1, 255)
    IM = cv2.erode(niblack(im, window_size=61, k=0.2), rect33)
    inpainted, modified = inpaint.inpaint_ng14(im, -IM)
    im_bg = (inpainted & ~IM) | (modified & IM)
    im_bg = im_bg.astype(np.float32).clip(0.1, 255)
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

def adaptive_otsu(im):
    im_h, _ = im.shape
    s = (im_h / 200) | 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    background = cv2.morphologyEx(im, cv2.MORPH_DILATE, ellipse)
    bg_float = background.astype(np.float64)
    debug_imwrite('bg.png', background)
    C = np.percentile(im, 30)
    normalized = clip_u8(C / (bg_float + 1e-10) * im)
    debug_imwrite('norm.png', normalized)
    return otsu(normalized)

def su2013(im, gamma=0.25):
    W = 5
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (W, 1))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (1, W))
    I_min = cv2.erode(cv2.erode(im, horiz), vert)
    I_max = cv2.dilate(cv2.dilate(im, horiz), vert)
    diff = I_max - I_min
    C = diff.astype(np.float32) / (I_max + I_min + 1e-16)

    alpha = (im.std() / 128.0) ** gamma
    print 'alpha:', alpha
    C_a = alpha * C + (1 - alpha) * diff

    _, C_a_bw = cv2.threshold(C_a, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # TODO: finish
    return C_a_bw

def binarize(im, algorithm=adaptive_otsu, gray=CIELab_gray, resize=1.0):
    if (im + 1 < 2).all():  # black and white
        return im
    else:
        if resize < 0.99 or resize > 1.01:
            im = cv2.resize(im, (0, 0), None, resize, resize)
        if len(im.shape) > 2:
            lum = gray(im)
            return algorithm(lum)  # & yan(l, T=35)
        else:
            return algorithm(im)

def go(argv):
    im = cv2.imread(argv[1], cv2.IMREAD_UNCHANGED)
    lib.debug = True
    lib.debug_prefix = 'ng2014/'
    cv2.imwrite('ng2014.png', binarize(im, algorithm=ntirogiannis2014))
    # cv2.imwrite('ng2014_hls.png', binarize(im, algorithm=ntirogiannis2014,
    #                                        gray=hls_gray))
    cv2.imwrite('ng2014_pca.png', binarize(im, algorithm=ntirogiannis2014,
                                           gray=pca_gray))
    lib.debug_prefix = 'yan/'
    cv2.imwrite('yan.png', binarize(im, algorithm=yan))
    lib.debug_prefix = 'lu2010/'
    cv2.imwrite('lu2010.png', binarize(im, algorithm=lu2010))

if __name__ == '__main__':
    go(sys.argv)
