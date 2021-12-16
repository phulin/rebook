from __future__ import division, print_function

import cv2
import numpy as np

from dewarp import algorithm, collate, lib


def split_lines(lines, all_lines=None):
    # Maximize horizontal separation
    # sorted by starting x value, ascending).
    lines = sorted(lines[:], key=lambda line: line.left())

    # Greedy algorithm. Maximize L bound of R minus R bound of L.
    current_r = 0
    quantity = -100000
    argmax = -1
    for idx, line in enumerate(lines[2:-3], 2):
        if line.right() - line.left() > 2000:
            print("line:", line.crop())
            for l in line:
                print(l)
        current_r = max(current_r, line.right())
        x2 = lines[idx + 1].left()
        if lib.debug:
            pass
            # print('x2:', x2, 'r:', current_r, 'quantity:', x2 - current_r)
        if x2 - current_r > quantity:
            quantity = x2 - current_r
            argmax = idx

    if lib.debug:
        print("split:", argmax, "out of", len(lines), "@", current_r)

    line_groups = [l for l in (lines[: argmax + 1], lines[argmax + 1 :]) if l]

    if all_lines is None:
        return line_groups
    else:
        if len(line_groups) == 1:
            return line_groups, [all_lines]
        else:
            boundary = (lines[argmax].left() + lines[argmax + 1].left()) / 2
            all_lines = sorted(all_lines, key=lambda line: line.left())
            lefts = [line.left() for line in all_lines]
            middle_idx = np.searchsorted(lefts, boundary)
            print("boundary:", boundary)
            print("left:", lefts[:middle_idx])
            print("right:", lefts[middle_idx:])
            return line_groups, [all_lines[:middle_idx], all_lines[middle_idx:]]


def filter_position(AH, im, lines, split):
    new_lines = []

    line_lefts = np.array([line.left() for line in lines])
    line_rights = np.array([line.right() for line in lines])
    line_start_thresh = np.percentile(line_lefts, 15 if split else 30) - 15 * AH
    line_end_thresh = np.percentile(line_rights, 85 if split else 70) + 15 * AH

    debug = cv2.cvtColor(im, cv2.COLOR_GRAY2RGB)
    for line in lines:
        if line.right() < line_start_thresh or line.left() > line_end_thresh:
            line.crop().draw(debug, color=lib.RED)
        else:
            line.crop().draw(debug, color=lib.GREEN)
            new_lines.append(line)

    lib.debug_imwrite("position_filter.png", debug)

    return new_lines


def crop(im, bw, split=True):
    im_h, im_w = im.shape[:2]

    all_letters = algorithm.all_letters(bw)
    AH = algorithm.dominant_char_height(bw, letters=all_letters)
    letters = algorithm.filter_size(AH, bw, letters=all_letters)
    all_lines = collate.collate_lines(AH, letters)
    combined = algorithm.combine_underlined(AH, bw, all_lines, all_letters)
    lines = algorithm.remove_stroke_outliers(bw, combined)

    if not lines:
        print("WARNING: no lines in image.")
        return AH, []

    lines = filter_position(AH, bw, lines, split)
    lines = [line for line in lines if not np.all(line.crop().apply(bw) == 255)]

    if not lines:
        print("WARNING: eliminated all lines.")
        return AH, []

    if split and im_w > im_h:  # two pages
        line_sets = split_lines(lines)
    else:
        line_sets = [lines]

    return AH, line_sets
