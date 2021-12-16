from __future__ import print_function

import argparse
import cv2
import glob
import numpy as np
import os
import re
import sys
from fpdf import FPDF
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from os.path import join, isfile
from subprocess import check_call, check_output

import algorithm
import binarize
import dewarp
from crop import crop
from geometry import Crop
from lib import debug_imwrite
import lib

extension = '.png'
def process_image(original, dpi=None):
    original_rot90 = original

    for i in range(args.rotate // 90):
        original_rot90 = np.rot90(original_rot90)

    # original_rot90 = cv2.resize(original_rot90, (0, 0), None, 1.5, 1.5)
    im_h, im_w = original_rot90.shape[:2]
    # image height should be about 10 inches. round to 100
    if not dpi:
        dpi = int(round(im_h / 1100.0) * 100)
        print('detected dpi:', dpi)

    split = im_w > im_h # two pages

    cropped_images = []
    if args.dewarp:
        lib.debug_prefix.append('dewarp')
        dewarped_images = dewarp.kim2014(original_rot90)
        for im in dewarped_images:
            bw = binarize.binarize(im, algorithm=binarize.sauvola, resize=1.0)
            lib.debug_prefix.append('crop')
            _, [lines] = crop(im, bw, split=False)
            lib.debug_prefix.pop()
            c = Crop.from_lines(lines)
            if c.nonempty():
                cropped_images.append(Crop.from_whitespace(bw).apply(im))
        lib.debug_prefix.pop()
    else:
        bw = binarize.binarize(original_rot90, algorithm=binarize.adaptive_otsu, resize=1.0)
        debug_imwrite('thresholded.png', bw)
        AH, line_sets = crop(original_rot90, bw, split=split)

        for lines in line_sets:
            c = Crop.from_lines(lines)
            if c.nonempty():
                lib.debug = False
                bw_cropped = c.apply(bw)
                orig_cropped = c.apply(original_rot90)
                angle = algorithm.skew_angle(bw_cropped, original_rot90, AH, lines)
                if not np.isfinite(angle): angle = 0.
                rotated = algorithm.safe_rotate(orig_cropped, angle)

                rotated_bw = binarize.binarize(rotated, algorithm=binarize.adaptive_otsu)
                _, [new_lines] = crop(rotated, rotated_bw, split=False)

                # dewarped = algorithm.fine_dewarp(rotated, new_lines)
                # _, [new_lines] = crop(rotated, rotated_bw, split=False)
                new_crop = Crop.union_all([line.crop() for line in new_lines])

                if new_crop.nonempty():
                    # cropped = new_crop.apply(dewarped)
                    cropped = new_crop.apply(rotated)
                    cropped_images.append(cropped)

    out_images = []
    lib.debug_prefix.append('binarize')
    for i, cropped in enumerate(cropped_images):
        lib.debug_prefix.append('page{}'.format(i))
        if lib.is_bw(original_rot90):
            out_images.append(binarize.otsu(cropped))
        else:
            out_images.append(
                binarize.ng2014_fallback(binarize.grayscale(cropped))
            )
        lib.debug_prefix.pop()
    lib.debug_prefix.pop()

    return dpi, out_images

def process_file(file_args):
    (inpath, outdir, dpi) = file_args
    outfiles = glob.glob('{}/{}_*{}'.format(outdir, inpath[:-4], extension))
    if outfiles:
        print('skipping', inpath)
        return outfiles
    else:
        print('processing', inpath)

    original = lib.imread(inpath)
    dpi, out_images = process_image(original, dpi=dpi)
    for idx, outimg in enumerate(out_images):
        outfile = '{}/{}_{}{}'.format(outdir, inpath[:-4], idx, extension)
        print('    writing', outfile)
        cv2.imwrite(outfile, outimg)
        outfiles.append(outfile)

    return outfiles

def pdfimages(pdf_filename):
    assert pdf_filename.endswith('.pdf')
    dirpath = pdf_filename[:-4]
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)
    if not os.listdir(dirpath):
        check_call(['pdfimages', '-png', pdf_filename, join(dirpath, 'page')])
    return dirpath

def unzip(zip_filename):
    assert zip_filename.endswith('.zip')
    directory = os.path.dirname(zip_filename)
    filenames = check_output(['zipinfo', '-1', zip_filename]).splitlines()
    paths = [os.path.join(directory, f) for f in filenames]
    if not any([os.path.isfile(f) for f in paths]):
        check_call(['unzip', zip_filename])
    return [f for f in paths if os.path.isfile(f)]

def sorted_numeric(strings):
    return sorted(strings, key=lambda f: list(map(int, re.findall('[0-9]+', f))))

def accumulate_paths(target, accum):
    for path in target:
        assert os.path.exists(path)
        if os.path.isfile(path):
            if path.endswith('.pdf'):
                accumulate_paths([pdfimages(path)], accum)
            elif path.endswith('.zip'):
                accumulate_paths(unzip(path), accum)
            elif re.match(r'.*\.(png|jpg|tif|dng)', path):
                accum.append(path)
        else:
            assert os.path.isdir(path)
            files = [os.path.join(path, base) for base in sorted_numeric(os.listdir(path))]
            accumulate_paths(files, accum)

def run(args):
    if args.single_file:
        lib.debug = True
        im = lib.imread(args.single_file)
        _, out_images = process_image(im, dpi=args.dpi)
        for idx, outimg in enumerate(out_images):
            cv2.imwrite('out{}.png'.format(idx), outimg)
        return

    if args.concurrent:
        pool = Pool(cpu_count())
        map_fn = pool.map
    else:
        map_fn = map

    files = []
    accumulate_paths(args.indirs, files)
    files = sorted_numeric(list(set(files)))
    print('Files:', files)

    for p in files:
        d = os.path.dirname(p)
        if not os.path.isdir(join(args.outdir, d)):
            os.makedirs(join(args.outdir, d))

    outfiles = map_fn(process_file, list(zip(files,
                        [args.outdir] * len(files),
                        [args.dpi] * len(files))))

    outfiles = sum(outfiles, [])
    outfiles.sort(key=lambda f: list(map(int, re.findall('[0-9]+', f))))

    # outtif = join(args.outdir, 'out.tif')
    outpdfpath = join(args.outdir, 'out.pdf')
    if not isfile(outpdfpath):
        print('making pdf:', outpdfpath)
        pdf = FPDF(unit='in', format='Letter')
        pdf.set_margins(0, 0, 0)
        for outfile in outfiles:
            print('.', end='')
            sys.stdout.flush()
            pdf.add_page()
            im = cv2.imread(outfile)
            inches_w = float(im.shape[1]) / args.dpi
            inches_h = float(im.shape[0]) / args.dpi
            x = (8.5 - inches_w) / 2
            y = (11.0 - inches_h) / 2
            pdf.image(outfile, w=inches_w, x=x, y=y)
        print()

        pdf.output(name=outpdfpath)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Batch-process for PDF')
    parser.add_argument('outdir', nargs='?', help="Output directory")
    parser.add_argument('indirs', nargs='+', help="Input directory")
    parser.add_argument('-f', '--file', dest='single_file', action='store',
                        help="Run on single file instead")
    parser.add_argument('-c', '--concurrent', action='store_true',
                        help="Run w/ threads.")
    parser.add_argument('-d', '--dpi', action='store', type=int,
                        help="Force a particular DPI")
    parser.add_argument('--dewarp', action='store_true', help="Dewarp pages.")
    parser.add_argument('--rotate', action='store', type=int, choices=[0, 90, 180, 270],
                        default=0, help="Rotate CCW by 90, 180, or 270 degrees.")

    global args
    args = parser.parse_args()
    run(args)
