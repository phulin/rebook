from __future__ import print_function

import cv2
import numpy as np
import numpy.polynomial.polynomial as poly
import sys

from rebook import algorithm, inpaint, lib

from rebook.algorithm import fast_stroke_width
from rebook.lib import mean_std, normalize_u8, clip_u8, bool_to_u8, debug_imwrite

cross33 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
rect33 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
rect55 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


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
    lib.debug_imwrite("pca.png", result)
    return result


def otsu(im):
    _, thresh = cv2.threshold(im, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    debug_imwrite("otsu.png", thresh)
    return thresh


def teager(im):
    im32 = im.astype(np.int32)
    padded = np.pad(im32, (1, 1), "edge")

    return normalize_u8(
        3 * im32 ** 2
        - padded[2:, 2:] * padded[:-2, :-2] / 2
        - padded[2:, :-2] * padded[:-2, 2:] / 2
        - padded[2:, 1:-1] * padded[:-2, 1:-1]
        - padded[1:-1, 2:] * padded[1:-1, :-2]
    )


def ng2014_normalize(im):
    IM = niblack(im, window_size=61, k=-0.2)
    debug_imwrite("niblack.png", IM)
    IM = cv2.erode(IM, rect33)
    debug_imwrite("dilated.png", IM)

    inpainted_min, inpainted_avg, modified = inpaint.inpaint_ng14(im, -IM)
    debug_imwrite("inpainted_min.png", inpainted_min)
    debug_imwrite("inpainted_avg.png", inpainted_avg)

    bg = (inpainted_min & ~IM) | (modified & IM)
    debug_imwrite("bg.png", bg)
    bgp = (inpainted_avg & ~IM) | (modified & IM)
    debug_imwrite("bgp.png", bg)

    im_f = im.astype(float) + 1
    bg_f = bg.astype(float) + 1
    F = im_f / bg_f
    N = clip_u8(255 * (F - F.min()))
    debug_imwrite("N.png", N)

    return N, bgp


class HeightMap(object):
    def __init__(self, letters):
        self.letters = sorted(letters, key=lambda l: l.h)

        idx = 0
        unique_values, unique_indices = np.unique(
            [l.h for l in self.letters], return_index=True
        )

        # map from height -> start of range containing height in letters
        self.start_indices = [0]
        for idx, v1, v2 in zip(
            unique_indices, np.concatenate([[0], unique_values]), unique_values
        ):
            self.start_indices.extend([idx] * (v2 - v1))

        self.start_indices.append(len(self.letters))

        self.total_area = sum((letter.area() for letter in self.letters))

    def max_height(self):
        return self.letters[-1].h

    def height_area(self, height):
        return sum((letter.area() for letter in self[height]))

    # RC_j in paper
    def ratio_components(self, height):
        return float(len(self[height])) / len(self.letters)

    # RP_j in paper
    def ratio_pixels(self, height):
        return float(self.height_area(height)) / self.total_area

    def __getitem__(self, height):
        idx1 = self.start_indices[height]
        idx2 = self.start_indices[height + 1]
        return self.letters[idx1:idx2]


def skeleton(im):
    im_inv = ~im
    eroded = im_inv.copy()
    skel = np.zeros(im.shape, dtype=im.dtype)
    while eroded.max() == im_inv.max():  # presumably 255
        skel |= eroded & ~cv2.morphologyEx(eroded, cv2.MORPH_OPEN, cross33)
        eroded = cv2.erode(eroded, cross33)

    return ~skel


def gradient(im):
    im_inv = ~im
    closed = cv2.morphologyEx(im_inv, cv2.MORPH_CLOSE, rect33)
    opened = cv2.morphologyEx(im_inv, cv2.MORPH_OPEN, rect33)
    assert (closed >= opened).all()
    return closed - opened


def erode_square(im, size):
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    return cv2.erode(cv2.erode(im, horiz), vert)


def dilate_square(im, size):
    horiz = cv2.getStructuringElement(cv2.MORPH_RECT, (1, size))
    vert = cv2.getStructuringElement(cv2.MORPH_RECT, (size, 1))

    return cv2.dilate(cv2.dilate(im, horiz), vert)


def gradient2(im):
    im_inv = ~im
    mins = erode_square(im_inv, 15)
    maxes = dilate_square(im_inv, 15)

    diff = maxes - mins

    return bool_to_u8(diff < 50)


def sauvola_noisy(im, *args, **kwargs):
    return sauvola(im, *args, **kwargs) | gradient2(im)


# @lib.timeit
def ntirogiannis2014(im):
    lib.debug_prefix.append("ng2014")

    debug_imwrite("input.png", im)
    im_h, _ = im.shape
    N, BG_prime = ng2014_normalize(im)
    O = otsu(N)

    debug_imwrite("O.png", O)
    letters = algorithm.all_letters(O)
    height_map = HeightMap(letters)

    ratio_sum = 0
    for h in range(1, height_map.max_height() + 1):
        if len(height_map[h]) == 0:
            continue
        ratio_sum += height_map.ratio_pixels(h) / height_map.ratio_components(h)
        if ratio_sum > 1:
            break

    min_height = h

    if lib.debug:
        print("Accept components only >= height", h)

    OP = O.copy()
    for h in range(1, min_height):
        for letter in height_map[h]:
            sliced = letter.slice(OP)
            np.place(sliced, letter.raster(), 255)
    debug_imwrite("OP.png", OP)

    strokes = fast_stroke_width(OP)
    debug_imwrite("strokes.png", normalize_u8(strokes.clip(0, 10)))
    SW = int(round(strokes.sum() / np.count_nonzero(strokes)))
    if lib.debug:
        print("SW =", SW)

    S = skeleton(OP)
    debug_imwrite("S.png", S)

    S_inv = ~S
    # S_inv_32 = S_inv.astype(np.int32)

    # FG_count = np.count_nonzero(S_inv)
    FG_pos = im[S_inv.astype(bool)]
    FG_avg = FG_pos.mean()
    FG_std = FG_pos.std()
    # FG = (S_inv & im).astype(np.int32)
    # FG_avg = FG.sum() / float(FG_count)
    # FG_std = np.sqrt(((S_inv_32 & (FG - FG_avg)) ** 2).sum() / float(FG_count))
    if lib.debug:
        print("FG:", FG_avg, FG_std)

    BG_avg = BG_prime.mean()
    BG_std = BG_prime.std()
    if lib.debug:
        print("BG:", BG_avg, BG_std)

    if FG_avg + FG_std != 0:
        C = -50 * np.log10((FG_avg + FG_std) / (BG_avg - BG_std))
        k = -0.2 - 0.1 * C / 10
    else:  # This is the extreme case when the FG is 100% black, check the article explaination page before equation 5
        C = -50 * np.log10((2.5) / (BG_avg - BG_std))
        k = -0.2 - 0.1 * C / 10

    if lib.debug:
        print("niblack:", C, k)
    local = niblack(N, window_size=(2 * SW) | 1, k=k)
    debug_imwrite("local.png", local)
    local_CCs = algorithm.all_letters(local)

    # NB: paper uses OP here, which results in neglecting all small components.
    O_inv = ~O
    O_inv_32 = (
        O_inv.astype(np.int8, copy=False).astype(np.int32).astype(np.uint32, copy=False)
    )
    label_map_O_inv = O_inv_32 & local_CCs[0].label_map
    CO_inv = np.zeros(im.shape, dtype=np.uint8)
    for cc in local_CCs:
        if (
            np.count_nonzero(cc.slice(label_map_O_inv) == cc.label) / float(cc.area())
            >= C / 100
        ):
            CO_sliced = cc.slice(CO_inv)
            np.place(CO_sliced, cc.raster(), 255)

    CO = ~CO_inv
    debug_imwrite("CO.png", CO)

    CO_inv_dilated = cv2.dilate(CO_inv, rect33)
    FB = ~(CO_inv | ((~O) & CO_inv_dilated))
    debug_imwrite("FB.png", FB)

    lib.debug_prefix.pop()

    return FB


# Sometimes ng2014 returns bad results with tons of black pixels.
# Fall back to sauvola in that case.
def ng2014_fallback(im):
    result = ntirogiannis2014(im)
    if result.mean() > 180:
        return result
    else:
        return sauvola(im)


# @lib.timeit
def niblack(im, window_size=61, k=0.2):
    means, stds = mean_std(im, window_size)
    thresh = means + k * stds

    return bool_to_u8(im > thresh)


# @lib.timeit
def sauvola(im, window_size=61, k=0.2):
    assert im.dtype == np.uint8
    means, stds = mean_std(im, window_size)
    thresh = means * (1 + k * ((stds / 127) - 1))

    return bool_to_u8(im > thresh)


def kittler(im):
    h, g = np.histogram(im.ravel(), 256, [0, 256])
    h = h.astype(np.float)
    g = g.astype(np.float)
    g = g[:-1]
    c = np.cumsum(h)
    m = np.cumsum(h * g)
    s = np.cumsum(h * g ** 2)
    sigma_f = np.sqrt(s / c - (m / c) ** 2)
    cb = c[-1] - c
    mb = m[-1] - m
    sb = s[-1] - s
    sigma_b = np.sqrt(sb / cb - (mb / cb) ** 2)
    p = c / c[-1]
    v = (
        p * np.log(sigma_f)
        + (1 - p) * np.log(sigma_b)
        - p * np.log(p)
        - (1 - p) * np.log(1 - p)
    )
    v[~np.isfinite(v)] = np.inf
    idx = np.argmin(v)
    t = g[idx]
    _, thresh = cv2.threshold(im, t, 255, cv2.THRESH_BINARY)
    return thresh


def roth(im, s=51, t=0.8):
    im_h, im_w = im.shape
    means = cv2.blur(im, (s, s))
    ints = cv2.bitwise_not(bool_to_u8(im > means * t))
    debug_imwrite("roth.png", ints)
    return ints


# s = stroke width
def kamel(im, s=None, T=25):
    im_h, im_w = im.shape
    if s is None or s <= 0:
        s = im_h / 200
    size = 2 * s + 1
    means = cv2.blur(im, (size, size), borderType=cv2.BORDER_REFLECT)
    padded = np.pad(means, (s, s), "edge")
    im_plus_T = im.astype(np.int64) + T
    im_plus_T = im_plus_T.clip(min=0, max=255).astype(np.uint8)
    L1 = padded[0:im_h, 0:im_w] <= im_plus_T
    L2 = padded[0:im_h, s : im_w + s] <= im_plus_T
    L3 = padded[0:im_h, 2 * s : im_w + 2 * s] <= im_plus_T
    L4 = padded[s : im_h + s, 2 * s : im_w + 2 * s] <= im_plus_T
    L5 = padded[2 * s : im_h + 2 * s, 2 * s : im_w + 2 * s] <= im_plus_T
    L6 = padded[2 * s : im_h + 2 * s, s : im_w + s] <= im_plus_T
    L7 = padded[2 * s : im_h + 2 * s, 0:im_w] <= im_plus_T
    L0 = padded[s : im_h + s, 0:im_w] <= im_plus_T
    L04, L15, L26, L37 = L0 & L4, L1 & L5, L2 & L6, L3 & L7
    b = (L04 & L15) | (L15 & L26) | (L26 & L37) | (L37 & L04)

    return b.astype(np.uint8) * 255


def row_zero_run_lengths(row):
    bounded = np.hstack(([255], row, [255]))
    diffs = np.diff(bounded)
    (run_starts,) = np.where(diffs < 0)
    (run_ends,) = np.where(diffs > 0)
    return run_ends - run_starts


def horiz_zero_run_lengths(im):
    return np.hstack(list(map(row_zero_run_lengths, im)))


def yan(im, alpha=0.4):
    im_h, im_w = im.shape
    first_pass = adaptive_otsu(im)

    horiz_runs = horiz_zero_run_lengths(first_pass)
    vert_runs = horiz_zero_run_lengths(first_pass.T)
    run_length_hist, _ = np.histogram(
        np.hstack((horiz_runs, vert_runs)), bins=np.arange(0, im_h / 100)
    )
    argmax = run_length_hist.argmax()
    (candidates,) = np.where(
        run_length_hist[argmax : argmax + 3] > run_length_hist[argmax] * 0.8
    )
    stroke_width = candidates.max() + argmax
    print("stroke width:", stroke_width)

    size = 2 * stroke_width + 1
    means = cv2.blur(im, (size, size), borderType=cv2.BORDER_REFLECT)

    element = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    maxes = cv2.morphologyEx(im, cv2.MORPH_DILATE, element).astype(float)
    mins = cv2.morphologyEx(im, cv2.MORPH_ERODE, element).astype(float)
    # if means closer to max, probably more noisy gray levels
    assert (maxes >= means).all() and (means >= mins).all()
    Ts = np.where(
        maxes - means > means - mins,
        alpha / 3 * (mins + mins + means),
        alpha / 3 * (mins + means + means),
    )

    result = kamel(im, s=stroke_width, T=Ts)
    return result


def median_downsample(row, step):
    (row_w,) = row.shape
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
    (row_w,) = row.shape

    step = row_w / 500  # target this many points in interpolation of rowage
    small = median_downsample(row, step)
    (small_w,) = small.shape
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
    (indices,) = np.where(im.flatten() > 0)
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
    debug_imwrite("bg.png", im_bg)

    C = np.percentile(im, 30)
    I_bar = clip_u8(C / im_bg * im)
    debug_imwrite("ibar.png", I_bar)
    I_bar_padded = np.pad(I_bar, (1, 1), "edge")

    V_h = cv2.absdiff(I_bar_padded[2:, 1:-1], I_bar_padded[:-2, 1:-1])
    V_v = cv2.absdiff(I_bar_padded[1:-1, 2:], I_bar_padded[1:-1, :-2])
    V = V_h + V_v
    V[V < V_h] = 255  # clip overflow
    _, stroke_edges = cv2.threshold(V, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # 1 if high contrast, 0 otherwise.
    E_inv = stroke_edges & 1
    E_inv_255 = E_inv * 255
    debug_imwrite("e_inv.png", E_inv_255)
    im_high = im & E_inv_255
    debug_imwrite("im_high.png", im_high)

    # calculate stroke width
    H, _ = np.histogram(nonzero_distances_row(E_inv), np.arange(im_h / 100))
    H[1] = 0  # don't use adjacent pix
    W = H.argmax()
    print("stroke width:", W)
    size = 2 * W
    N_min = W

    if size >= 16:
        E_inv = E_inv.astype(np.uint16)
    window = (size | 1, size | 1)
    N_e = cv2.boxFilter(E_inv, -1, window, normalize=False)
    debug_imwrite("n_e.png", bool_to_u8(N_e >= N_min))

    E_mean = cv2.boxFilter(im_high, cv2.CV_32S, window, normalize=False) / (
        N_e.astype(np.float64) + 0.1
    )
    debug_imwrite("i_bar_e_mean.png", bool_to_u8(I_bar <= E_mean))

    out = ~bool_to_u8((N_e >= N_min) & (I_bar <= E_mean))
    debug_imwrite("lu2010.png", out)
    return out


def adaptive_otsu(im):
    im_h, _ = im.shape
    s = (im_h // 200) | 1
    ellipse = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (s, s))
    background = cv2.morphologyEx(im, cv2.MORPH_DILATE, ellipse)
    bg_float = background.astype(np.float64)
    debug_imwrite("bg.png", background)
    C = np.percentile(im, 30)
    normalized = clip_u8(C / (bg_float + 1e-10) * im)
    debug_imwrite("norm.png", normalized)
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
    C_a = alpha * C + (1 - alpha) * diff
    _, C_a_bw = cv2.threshold(
        C_a.astype("uint8"), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )

    # TODO: finish
    return C_a_bw


def retinex(im, mu_1=0.9, mu_2=25, sigma=5):
    G = cv2.GaussianBlur(im, (0, 0), sigma)
    debug_imwrite("G.png", G)
    bools = (im < mu_1 * G) & (cv2.absdiff(im, G) > mu_2)
    return bool_to_u8(bools)


def premultiply(im):
    assert im.dtype == np.uint8
    im32 = im[:, :, :-1].astype(np.uint32)
    im32 *= im[:, :, -1:]
    im32 >>= 8
    return im32.astype(np.uint8)


def grayscale(im, algorithm=CIELab_gray):
    if len(im.shape) > 2:
        if im.shape[2] == 4:
            return algorithm(premultiply(im))
        else:
            return algorithm(im)
    else:
        return im


def binarize(im, algorithm=adaptive_otsu, gray=CIELab_gray, resize=1.0):
    if (im + 1 < 2).all():  # black and white
        return im
    else:
        if resize < 0.99 or resize > 1.01:
            im = cv2.resize(im, (0, 0), None, resize, resize)
        return algorithm(grayscale(im, algorithm=gray))


def go(argv):
    im = grayscale(lib.imread(argv[1]))
    lib.debug = True
    lib.debug_prefix = ["binarize"]
    # lib.debug_imwrite('gradient2.png', gradient2(im))
    lib.debug_imwrite("sauvola_noisy.png", sauvola_noisy(im, k=0.1))
    # lib.debug_imwrite('adaptive_otsu.png', binarize(im, algorithm=adaptive_otsu))
    # lib.debug_imwrite('ng2014.png', binarize(im, algorithm=ntirogiannis2014))
    # lib.debug_imwrite('yan.png', binarize(im, algorithm=yan))
    lib.debug_imwrite("sauvola.png", sauvola(im, k=0.1))
    # lib.debug_imwrite('retinex.png', binarize(im, algorithm=retinex))


if __name__ == "__main__":
    go(sys.argv)
