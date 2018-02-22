from __future__ import print_function

import argparse
import cv2
import glob
import os
import re
from multiprocessing import cpu_count
from multiprocessing.pool import Pool
from os.path import join, isfile
from subprocess import check_call

import algorithm
import binarize
from crop import crop
from geometry import Crop
from lib import debug_imwrite
import lib

extension = '.tif'
def process_image(original, dpi):
    # original = cv2.resize(original, (0, 0), None, 1.5, 1.5)
    im_h, im_w = original.shape
    # image height should be about 10 inches. round to 100
    if not dpi:
        dpi = int(round(im_h / 1100.0) * 100)
        print('detected dpi:', dpi)
    split = im_w > im_h # two pages

    bw = binarize.binarize(original, algorithm=binarize.adaptive_otsu, resize=1.0)
    debug_imwrite('thresholded.png', bw)
    AH, line_sets = crop(original, bw, split=split)

    outimgs = []
    for lines in line_sets:
        c = Crop.union_all([line.crop() for line in lines])
        if c.nonempty():
            bw_cropped = c.apply(bw)
            orig_cropped = c.apply(original)
            angle = algorithm.skew_angle(bw_cropped, original, AH, lines)
            rotated = algorithm.safe_rotate(orig_cropped, angle)

            rotated_bw = binarize.binarize(rotated, algorithm=binarize.adaptive_otsu, resize=1.0)
            _, [new_lines] = crop(rotated, rotated_bw, split=False)
            new_crop = Crop.union_all([line.crop() for line in new_lines])
            # algorithm.fine_dewarp(rotated, new_lines)
            lib.debug = False

            if new_crop.nonempty():
                cropped = new_crop.apply(rotated)
                if lib.is_bw(original):
                    outimgs.append(binarize.otsu(cropped))
                else:
                    outimgs.append(
                        binarize.ng2014_normalize(binarize.grayscale(rotated))
                    )

    return dpi, outimgs

def process_file(args):
    (inpath, outdir, dpi) = args
    outfiles = glob.glob('{}/{}_*{}'.format(outdir, inpath[:-4], extension))
    if outfiles:
        print('skipping', inpath)
        if dpi is not None:
            for outfile in outfiles:
                check_call(['tiffset', '-s', '282', str(dpi), outfile])
                check_call(['tiffset', '-s', '283', str(dpi), outfile])
        return outfiles
    else:
        print('processing', inpath)

    original = cv2.imread(inpath, cv2.IMREAD_UNCHANGED)
    dpi, outimgs = process_image(original, dpi)
    for idx, outimg in enumerate(outimgs):
        outfile = '{}/{}_{}{}'.format(outdir, inpath[:-4], idx, extension)
        print('    writing', outfile)
        cv2.imwrite(outfile, outimg)
        check_call(['tiffset', '-s', '282', str(dpi), outfile])
        check_call(['tiffset', '-s', '283', str(dpi), outfile])
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

def sorted_numeric(strings):
    return sorted(strings, key=lambda f: list(map(int, re.findall('[0-9]+', f))))

def accumulate_paths(target, accum):
    for path in target:
        if os.path.isfile(path):
            if path.endswith('.pdf'):
                accumulate_paths([pdfimages(path)], accum)
            elif re.match(r'.*\.(png|jpg|tif)', path):
                accum.append(path)
        else:
            assert os.path.isdir(path)
            files = [os.path.join(path, base) for base in sorted_numeric(os.listdir(path))]
            accumulate_paths(files, accum)

def run(args):
    if args.single_file:
        lib.debug = True
        im = cv2.imread(args.single_file, cv2.IMREAD_UNCHANGED)
        _, outimgs = process_image(im, args.dpi)
        for idx, outimg in enumerate(outimgs):
            cv2.imwrite('out{}.png'.format(idx), outimg)
        return

    if args.concurrent:
        pool = Pool(cpu_count())
        map_fn = pool.map
    else:
        map_fn = map

    files = []
    accumulate_paths(args.indirs, files)
    print(files)

    for p in files:
        d = os.path.dirname(p)
        if not os.path.isdir(join(args.outdir, d)):
            os.makedirs(join(args.outdir, d))

    outfiles = map_fn(process_file, list(zip(files,
                        [args.outdir] * len(files),
                        [args.dpi] * len(files))))

    outfiles = sum(outfiles, [])
    outfiles.sort(key=lambda f: list(map(int, re.findall('[0-9]+', f))))

    outtif = join(args.outdir, 'out.tif')
    outpdf = join(args.outdir, 'out.pdf')
    if not isfile(outpdf):
        if not isfile(outtif):
            print('making tif:', outtif)
            check_call(['tiffcp'] + outfiles + [outtif])

        print('making pdf:', outpdf)
        check_call([
            'tiff2pdf', '-z', '-p', 'letter',
            '-o', outpdf, outtif
        ])

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

    run(parser.parse_args())
