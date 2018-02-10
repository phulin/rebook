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

def draw_box(debug, c, color, thickness):
    if not lib.debug: return
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(debug, (x, y), (x + w, y + h), color, 4)

def crop(im, bw, split=True):
    im_h, im_w = im.shape

    AH = algorithm.dominant_char_height(bw)
    letter_boxes = algorithm.letter_contours(AH, bw)
    lines = collate.collate_lines(AH, letter_boxes)

    stroke_widths = algorithm.fast_stroke_width(bw)
    lib.debug_imwrite('strokes.png', lib.normalize_u8(stroke_widths.clip(0, 10)))

    mask = np.zeros(im.shape, dtype=np.uint8)
    letter_contours = [c for (c, _, _, _, _) in letter_boxes]
    cv2.drawContours(mask, letter_contours, -1, 255, thickness=cv2.FILLED)

    masked_strokes = np.ma.masked_where(mask ^ 255, stroke_widths)
    strokes_mean = masked_strokes.mean()
    strokes_std = masked_strokes.std()
    print('overall: mean:', strokes_mean, 'std:', strokes_std)

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    line_crops = []
    good_contours = []
    for line in lines:
        line_crop = Crop.null(bw)
        if len(line) <= 1: continue
        for c, x, y, w, h in line:
            crop = Crop.from_rect(x, y, w, h)

            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(mask, [c], 0, 255,
                             thickness=cv2.FILLED, offset=(-x, -y))
            masked_strokes = np.ma.masked_where(mask ^ 255,
                                                crop.apply(stroke_widths))
            # print 'mean:', masked_strokes.mean(), 'std:', masked_strokes.std()
            mean = masked_strokes.mean()
            if mean < strokes_mean - strokes_std:
                print('skipping{: 5d}{: 5d} {:.03f} {:.03f}'.format(
                    x, y, mean, masked_strokes.std()
                ))
                draw_box(debug, c, (0, 0, 255), 2)
            else:
                draw_box(debug, c, (0, 255, 0), 2)
                line_crop = line_crop.union(crop)
                good_contours.append(c)

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


