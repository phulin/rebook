from __future__ import division, print_function

import cv2
import numpy as np

import algorithm
import collate
from geometry import Crop
import lib

def draw_crop(im, crop, color, thickness=2):
    if not lib.debug: return
    cv2.rectangle(im, (crop.x0, crop.y0), (crop.x1, crop.y1), color, thickness)

def split_crops(crops):
    # Maximize horizontal separation
    # sorted by starting x value, ascending).
    crops = sorted(crops, key=lambda crop: crop.x0)

    # Greedy algorithm. Maximize L bound of R minus R bound of L.
    current_r = 0
    quantity = -100000
    argmax = -1
    for idx, crop in list(enumerate(crops))[2:-3]:
        current_r = max(current_r, crop.x1)
        x2 = crops[idx + 1].x0
        # print 'x2:', x2, 'r:', current_r, 'quantity:', x2 - current_r
        if x2 - current_r > quantity:
            quantity = x2 - current_r
            argmax = idx

    print('split:', argmax, 'out of', len(crops), '@', current_r)

    return [l for l in (crops[:argmax + 1], crops[argmax + 1:]) if l]

def masked_mean_std(data, mask):
    mask_sum = np.count_nonzero(mask)
    mean = data.sum() / mask_sum
    data = data.astype(np.float64, copy=False)
    data_dev = np.zeros(data.shape, dtype=np.float64)
    np.subtract(data, mean, out=data_dev, where=mask.astype(bool, copy=False))
    std = np.sqrt(np.square(data_dev).sum() / mask_sum)
    return mean, std

def crop(im, bw, split=True):
    im_h, im_w = im.shape

    all_letters = algorithm.all_letters(bw)
    AH = algorithm.dominant_char_height(bw, letters=all_letters)
    letters = algorithm.letter_contours(AH, bw, letters=all_letters)
    lines = collate.collate_lines(AH, letters)

    stroke_widths = algorithm.fast_stroke_width(bw)
    if lib.debug:
        lib.debug_imwrite('strokes.png', lib.normalize_u8(stroke_widths.clip(0, 10)))

    mask = np.zeros(im.shape, dtype=np.uint8)
    for line in lines:
        for letter in line:
            sliced = letter.crop().apply(mask)
            sliced += letter.raster()

    lib.debug_imwrite('letter_mask.png', -mask)

    masked_strokes = stroke_widths.copy()
    masked_strokes &= -mask

    strokes_mean, strokes_std = masked_mean_std(masked_strokes, mask)
    print('overall: mean:', strokes_mean, 'std:', strokes_std)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    line_crops = []
    for line in lines:
        line_crop = Crop.null(bw)
        if len(line) <= 1: continue
        for letter in line:
            crop = letter.crop()
            raster = letter.raster()
            sliced_strokes = crop.apply(stroke_widths).copy()
            sliced_strokes &= lib.bool_to_u8(raster)

            mean, std = masked_mean_std(sliced_strokes, raster)
            if mean < strokes_mean - strokes_std:
                print('skipping {:4d} {:4d} {:.03f} {:.03f}'.format(
                    letter.x, letter.y, mean, std,
                ))
                if lib.debug: letter.box(debug, color=lib.RED)
            else:
                if lib.debug: letter.box(debug, color=lib.GREEN)
                line_crop = line_crop.union(crop)

        line_crops.append(line_crop)

    if not line_crops:
        print('WARNING: no lines in image.')
        return AH, lines, []

    line_lefts = np.array([lc.x0 for lc in line_crops])
    line_rights = np.array([lc.x1 for lc in line_crops])
    line_start_thresh = np.percentile(line_lefts, 15 if split else 30)
    line_end_thresh = np.percentile(line_rights, 85 if split else 70)
    good_line_crops = []
    line_crop_debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for line_crop in line_crops:
        if (line_crop.x1 + 15 * AH < line_start_thresh or
            line_crop.x0 - 15 * AH > line_end_thresh):
            draw_crop(line_crop_debug, line_crop, (0, 0, 255))
        else:
            good_line_crops.append(line_crop)
            draw_crop(line_crop_debug, line_crop, (0, 255, 0))

    lib.debug_imwrite("line_debug.png", line_crop_debug)
    lib.debug_imwrite("debug.png", debug)

    line_crops = [lc for lc in good_line_crops if lc.nonempty() and \
                        not np.all(lc.apply(bw) == 255)]

    if not line_crops:
        return AH, lines, [Crop.full(im)]

    if split and im_w > im_h:  # two pages
        crop_sets = split_crops(line_crops)
    else:
        crop_sets = [line_crops]

    return AH, lines, [Crop.union_all(cs) for cs in crop_sets]
