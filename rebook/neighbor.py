from __future__ import print_function, division

import argparse
import cv2
import freetype
import joblib
import numpy as np
from numpy import dot
import os
import sklearn.decomposition

import algorithm
import binarize
import lib
from training import create_mosaic, patches, print_dict

def features_lo(lo_res):
    lo_res_hi = cv2.resize(lo_res, (0, 0), None, 2.0, 2.0,
                           interpolation=cv2.INTER_CUBIC)

    return lo_res_hi, np.stack([
        cv2.Sobel(lo_res_hi, cv2.CV_64F, 1, 0),
        cv2.Sobel(lo_res_hi, cv2.CV_64F, 0, 1),
        cv2.Laplacian(lo_res_hi, cv2.CV_64F),
    ], axis=2)

def training_data(font_paths, font_size, W_h):
    faces = [freetype.Face(font_path) for font_path in font_paths]

    hi_res = np.concatenate([create_mosaic(face, font_size) for face in faces])

    blurred_ims = [
        cv2.GaussianBlur(hi_res, (0, 0), 7, 3),
        cv2.GaussianBlur(hi_res, (0, 0), 3, 7),
    ]
    blurred = np.concatenate(blurred_ims, axis=0)
    hi_res_2 = np.tile(hi_res, (len(blurred_ims), 1))
    lib.debug_imwrite('hi.png', hi_res_2)

    lo_res = cv2.resize(blurred, (0, 0), None, 0.5, 0.5,
                        interpolation=cv2.INTER_AREA)
    lib.debug_imwrite('lo.png', lo_res)

    lo_res_hi, filtered_lo = features_lo(lo_res)

    difference = hi_res_2.astype(np.float64) - lo_res_hi
    lib.debug_imwrite('diff.png', lib.normalize_u8(difference))

    # make sure we're on edges (in hi-res reference)
    struct = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    gradient = cv2.morphologyEx(hi_res_2, cv2.MORPH_GRADIENT, struct)
    gradient_means, _ = lib.mean_std(gradient, W_h)
    patch_mask = gradient_means > np.percentile(gradient_means, 50)

    # patch_centers should match others' shape.
    step = 3
    center_slice = slice(W_h // 2, -(W_h // 2) - 1, step)
    patch_centers = patch_mask[center_slice, center_slice]
    lo_patches = patches(filtered_lo, W_h, step)[patch_centers].transpose(0, 3, 1, 2)
    hi_patches = patches(difference, W_h, step)[patch_centers]
    t = lo_patches.shape[0]

    lo_patches_vec = lo_patches.reshape(t, -1)
    for i in range(lo_patches.shape[1]):
        print_dict('lo_sq{}.png'.format(i),
                   lo_patches_vec[:, i * W_h * W_h:(i + 1) * W_h * W_h])
    hi_patches_vec = hi_patches.reshape(t, W_h * W_h)
    print_dict('hi_sq.png', hi_patches_vec)

    # reduce dimensionality on lo-res patches with PCA.
    pca = sklearn.decomposition.PCA(n_components=lo_patches_vec.shape[1] // 6)
    Y_pca = pca.fit_transform(lo_patches_vec)

    return Y_pca, hi_patches_vec, pca

def all_file(fns):
    return all([os.path.isfile(fn) for fn in fns])

def train(dest, font_path, sizes):
    K = 1024

    for size in sizes:
        W_l = size // 3
        W_h = 2 * W_l + 1

        dest_dir = os.path.join(dest, str(size))
        if not os.path.isdir(dest_dir):
            print('making directory', dest_dir)
            os.makedirs(dest_dir)

        training_lo_file = os.path.join(dest_dir, 'training_lo.npy')
        training_hi_file = os.path.join(dest_dir, 'training_hi.npy')
        training_pca_file = os.path.join(dest_dir, 'training_pca.pkl')
        dict_lo_file = os.path.join(dest_dir, 'dict_lo.npy')
        mapping_file = os.path.join(dest_dir, 'dict_lo_mapping.npy')
        dict_hi_file = os.path.join(dest_dir, 'dict_hi.npy')

        if all_file((training_lo_file, training_hi_file, training_pca_file)):
            P_l_T = np.load(training_lo_file)
            P_h_T = np.load(training_hi_file)
            pca = joblib.load(training_pca_file)
        else:
            P_l_T, P_h_T, pca = training_data(font_path, size * 2, W_h)
            np.save(training_lo_file, P_l_T)
            np.save(training_hi_file, P_h_T)
            joblib.dump(pca, training_pca_file)

        if all_file((dict_lo_file, mapping_file)):
            A_l_T = np.load(dict_lo_file)
            Q_T = np.load(mapping_file)
            print(Q_T.shape, A_l_T.shape)
        else:
            print('loaded. running K-SVD.')
            ksvd_model = ksvd.ApproximateKSVD(K)
            ksvd_model.fit(P_l_T)
            A_l_T = ksvd_model.components_
            Q_T = ksvd_model.transform(P_l_T)
            np.save(dict_lo_file, A_l_T)
            np.save(mapping_file, Q_T)

        if all_file((dict_hi_file,)):
            A_h_T = np.load(dict_hi_file)
        else:
            print('making hi dict.')
            A_h_T = np.linalg.solve(dot(Q_T.T, Q_T), dot(Q_T.T, P_h_T))
            np.save(dict_hi_file, A_h_T)

def load_model(model_dir):
    training_lo_file = os.path.join(model_dir, 'training_lo.npy')
    training_hi_file = os.path.join(model_dir, 'training_hi.npy')
    training_pca_file = os.path.join(model_dir, 'training_pca.pkl')
    dict_lo_file = os.path.join(model_dir, 'dict_lo.npy')
    mapping_file = os.path.join(model_dir, 'dict_lo_mapping.npy')
    dict_hi_file = os.path.join(model_dir, 'dict_hi.npy')

    return (
        np.load(training_lo_file),  # P_l_T
        np.load(training_hi_file),  # P_h_T
        joblib.load(training_pca_file),  # pca
        np.load(dict_lo_file),  # A_l_T
        np.load(mapping_file),  # Q_T
        np.load(dict_hi_file),  # A_h_T
    )

def test(models_dir, image_path):
    im = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

    bw = binarize.binarize(im, algorithm=binarize.ntirogiannis2014)
    AH = algorithm.dominant_char_height(bw)
    print('AH =', AH)

    possible_AHs = np.array([int(d) for d in os.listdir(models_dir) if d.isdigit()])
    size = possible_AHs[np.abs(possible_AHs - AH).argmin()]
    model_dir = os.path.join(models_dir, str(size))

    P_l_T, P_h_T, pca, A_l_T, Q_T, A_h_T = load_model(model_dir)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Upscale using A+.')
    parser.add_argument('model', action='store',
                        help='Directory for model storage.')
    parser.add_argument('--train', action='store_true',
                        help='Train model into specified directory.')
    parser.add_argument('--test', action='store',
                        help='Try upscaling an image by 2x.')
    args = parser.parse_args()

    lib.debug = True
    lib.debug_prefix = 'neighbor/'
    np.set_printoptions(precision=3, linewidth=80)

    if args.train:
        import ksvd
        # these sizes should correspond to AH in scanned stuff.
        sizes = [15, 18, 20, 22, 26, 30]
        fonts = [
            "/Library/Fonts/Microsoft/Constantia.ttf",
            "/Library/Fonts/Times New Roman.ttf",
        ]
        train(args.model, fonts, sizes)
    elif args.test:
        test(args.model, args.test)
