from __future__ import print_function, division

import cv2
import numpy as np
import scipy.spatial
import sys

import algorithm
import binarize
import lib
from lib import GREEN

N_values = np.array([64, 64, 64, 128, 128, 128, 256, 256, 256, 256])
k_values = np.array([5, 4, 3, 5, 4, 3, 5, 4, 3, 2])
s_values = N_values.astype(np.float64) / k_values

theta_values = np.arange(32) / np.pi

# radius of circle for "nearby" CCs projection
radius = 100
radius_sq = radius ** 2

epsilon = 2.8

def pack_label(s, theta):
    return theta * s_values.shape[0] + s

def unpack_label(label):
    s_len = s_values.shape[0]
    return label % s_len, label // s_len

def V_p(nearby_centroids, centroids_rotated, ellipses_sheared):
    result = np.zeros((centroids_rotated[0].shape[0],
                       theta_values.shape[0] * s_values.shape[0]),
                      dtype=np.float64)
    for theta, centroids in enumerate(centroids_rotated):
        for i, centroid in enumerate(centroids):
            nearby = nearby_centroids[i]
            if nearby.shape[0] <= 3:
                for s in range(s_values.shape[0]):
                    result[i, pack_label(s, theta)] = epsilon
                continue

            ellipses = ellipses_sheared[theta][nearby]

            box_corner = np.floor(centroid - radius)
            nearby_centroids = centroids_rotated[theta][nearby] - box_corner
            signal = np.zeros((2 * radius + 2,), dtype=np.float64)
            ys = np.arange(0, 2 * radius + 2)
            for centroid_2, (x_axis, y_axis) in zip(nearby_centroids, ellipses):
                x_0, y_0 = centroid_2
                xs_sq = x_axis ** 2 * (1 - ((ys - y_0) / y_axis) ** 2)
                signal += np.where(xs_sq >= 0, 2 * np.sqrt(xs_sq), 0)

            for s in range(s_values.shape[0]):
                N = N_values[s]
                k = k_values[s]

                t = signal.shape[0]
                signal_padded = np.pad(signal, (0, N - (t % N)),
                                    'constant', constant_values=0)
                signal_N = signal_padded.reshape(N, -1).sum(axis=1)
                X = np.fft.rfft(signal_N)
                result[i, pack_label(s, theta)] = \
                    -np.log(np.abs(X[k]) ** 2 / np.abs(X[0]) ** 2)

    return result

lam_1 = 0.5
lam_2 = 5
k = 0.125
def V_pq(s_x, theta_x, centroids, segments):
    s_p, theta_p = s_x[segments[:, 0]], theta_x[segments[:, 0]]
    s_q, theta_q = s_x[segments[:, 1]], theta_x[segments[:, 1]]
    centroids_p = centroids[segments[:, 0]]
    centroids_q = centroids[segments[:, 1]]

    d_pq_sq = np.square(centroids_p - centroids_q).sum(axis=1)
    scale = np.exp(-k * d_pq_sq / (np.square(s_values[s_p]) + np.square(s_values[s_q])))

    f_p = np.stack([s_p, theta_p])
    f_q = np.stack([s_q, theta_q])
    f_diff = np.abs(f_p - f_q).sum(axis=0)
    mu = np.where(f_p == f_q, 0, np.where(f_diff <= 3, lam_1, lam_2))

    return mu * scale

# im must be white-on-black.
def letter_ellipses(im):
    num_labels, labels, stats, all_centroids = cv2.connectedComponentsWithStats(im)
    print(labels)
    print(num_labels)

    boxes = stats[:, (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                      cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT)]

    centroids_list = []
    ellipses_list = []
    for i in range(1, num_labels):
        x, y, w, h = boxes[i]
        centroid = all_centroids[i]
        local = labels[y:y + h, x:x + w] == i
        points = np.array(np.nonzero(local))[::-1]  # coord order (x, y)
        covariance = np.cov(points)
        assert covariance.shape == (2, 2)

        # these are normalized to 1; normalize to sqrt(w)
        eigvals, eigvecs = np.linalg.eigh(covariance)
        sig_2, sig_1 = eigvals
        area = stats[i, cv2.CC_STAT_AREA]
        if sig_1 / sig_2 <= 15 and area > 10 and area < 3000:
            eigvecs *= np.sqrt(eigvals)
            centroids_list.append(i)
            ellipses_list.append(eigvecs)

    centroids = all_centroids[centroids_list]
    ellipses = np.array(ellipses_list)

    if lib.debug:
        debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        for centroid, eigvecs in zip(centroids, ellipses):
            for v in eigvecs.T:
                cv2.line(debug, tuple((centroid + v).astype(int)),
                        tuple((centroid - v).astype(int)), GREEN, 2)
        lib.debug_imwrite('ellipses.png', debug)

    return centroids, ellipses

def precompute_rotations(im, centroids, ellipses):
    # precompute a bunch of stuff.
    im_h, im_w = im.shape
    diag = np.sqrt(im_h ** 2 + im_w ** 2)
    centroids_rotated = []
    ellipses_sheared = []
    padding_h = int(np.ceil((diag - im_h) / 2))
    padding_w = int(np.ceil((diag - im_w) / 2))
    centroids_safe = centroids + np.array((padding_w, padding_h))
    centroids_homo = np.concatenate([
        centroids_safe.T,
        np.ones((1, centroids_safe.shape[0]))
    ])
    new_h, new_w = im_h + 2 * padding_h, im_w + 2 * padding_w
    for i, theta in enumerate(theta_values):
        theta_deg = theta * 180 / np.pi
        matrix = cv2.getRotationMatrix2D((new_w / 2., new_h / 2.), theta_deg, 1)

        centroids_rotated.append(matrix.dot(centroids_homo).T)
        # FIXME: something is wrong with these rotations
        ellipse_rotated = matrix[:, :2].dot(ellipses.T).T

        if lib.debug and i == 5:
            im_safe = np.pad(im, ((padding_h, padding_h), (padding_w, padding_w)),
                            'constant', constant_values=0)
            im_rotated = cv2.warpAffine(im_safe, matrix, (new_w, new_h),
                                        borderMode=cv2.BORDER_CONSTANT,
                                        borderValue=0)
            debug = cv2.cvtColor(im_rotated, cv2.COLOR_GRAY2BGR)
            for centroid, eigvecs in zip(centroids_rotated[-1], ellipse_rotated):
                for v in eigvecs.T:
                    cv2.line(debug, lib.int_tuple(centroid + v),
                            lib.int_tuple(centroid - v), GREEN, 2)
            lib.debug_imwrite('ellipses_rot.png', debug)

        # shear ellipses in x-direction to make perp to y axis
        x1s, x2s = ellipse_rotated[0]
        y1s, y2s = ellipse_rotated[1]
        y0s = np.sqrt(y1s ** 2 + y2s ** 2)
        x0s = (x1s ** 2 + y1s * x2s) / np.sqrt(x1s ** 2 + y1s ** 2)
        ellipses_sheared.append(np.stack([x0s, y0s]).T)

    return centroids_rotated, ellipses_sheared

def koo2010(im_inv, AH):
    im = -im_inv
    assert lib.is_bw(im)

    centroids, ellipses = letter_ellipses(im)
    centroids_rotated, ellipses_sheared = \
        precompute_rotations(im, centroids, ellipses)

    nearby_centroids = []
    for centroid in centroids:
        distances_sq = np.square(centroids - centroid).sum()
        nearby = (distances_sq < radius_sq) & ~np.all(centroids == centroid, axis=1)
        nearby_centroids.append(np.nonzero(nearby)[0])

    tri = scipy.spatial.Delaunay(centroids)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    for simplex in tri.simplices:
        for i, j in zip(simplex, np.roll(simplex, 1)):
            cv2.line(debug, tuple(centroids[i].astype(int)),
                        tuple(centroids[j].astype(int)), GREEN, 2)
    lib.debug_imwrite('triang.png', debug)

    duplicate_segments = np.concatenate([
        tri.simplices[:, (0, 1)],
        tri.simplices[:, (1, 2)],
        tri.simplices[:, (2, 0)],
    ])
    unordered_segments = np.unique(duplicate_segments, axis=0)
    segments = np.stack([
        unordered_segments.min(axis=1),
        unordered_segments.max(axis=1),
    ], axis=1)
    assert np.all(segments[:, 0] < segments[:, 1])

    theta_p = np.zeros((len(ellipses),))
    s_p = np.full((len(ellipses),), 5)

    def V_pq_sites(p1, p2, l1, l2):
        s_1, theta_1 = unpack_label(l1)
        s_2, theta_2 = unpack_label(l2)

        centroid_1, centroid_2 = centroids[(p1, p2), :]

        d_pq_sq = np.square(centroid_1 - centroid_2).sum()
        scale = np.exp(-k * d_pq_sq / (s_values[s_1] ** 2 + s_values[s_2] ** 2))

        f_diff = abs(s_1 - s_2) + abs(theta_1 - theta_2)
        mu = 0 if l1 == l2 else (lam_1 if f_diff <= 3 else lam_2)

        return mu * scale

    # graph.set_smooth_cost_function(V_pq_sites)
    # graph.expansion()

if __name__ == '__main__':
    lib.debug = True
    orig = cv2.imread(sys.argv[1])
    bw = binarize.binarize(orig, algorithm=binarize.sauvola)
    lib.debug_imwrite('retinex.png', bw)
    AH = algorithm.dominant_char_height(bw)
    koo2010(bw, AH)
