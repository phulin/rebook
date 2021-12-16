from __future__ import division, print_function

import cv2
import numpy as np
import os
import sys

import algorithm
import lib

hi_dir = sys.argv[1]
in_dir = sys.argv[2]
out_dir = sys.argv[3]

N_IMG = 1000
theta_range = [-np.pi / 45, np.pi / 45]  # +/- 4 deg
KERNELS = [(1, 1), (3, 3), (5, 1), (3, 7)]
NOISE = 15

imgs_base = [fn for fn in os.listdir(sys.argv[1]) if fn.endswith('.png')]
for i in range(N_IMG):
    fn_base = np.random.choice(imgs_base)
    print(i, fn_base)
    im = cv2.imread(os.path.join(hi_dir, fn_base), cv2.IMREAD_UNCHANGED)

    theta = (theta_range[1] - theta_range[0]) * np.random.random() \
        + theta_range[0]

    rotated = algorithm.safe_rotate(im, theta)
    cv2.imwrite(os.path.join(out_dir, 'im{}.png'.format(i)), rotated)

    kernel_std = KERNELS[np.random.choice(len(KERNELS))]
    blurred = cv2.GaussianBlur(rotated, (0, 0), kernel_std[0], kernel_std[1])
    noisy = lib.clip_u8(blurred.astype(np.float64) + 10 * np.random.randn(*blurred.shape))
    downsampled = cv2.resize(noisy, (0, 0), None, 0.5, 0.5,
                             interpolation=cv2.INTER_AREA)
    # _, binarized = cv2.threshold(downsampled, 140, 255, cv2.THRESH_BINARY)
    cv2.imwrite(os.path.join(in_dir, 'im{}.png'.format(i)), downsampled)
