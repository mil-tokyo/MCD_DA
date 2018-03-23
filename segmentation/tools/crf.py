# coding: utf-8
import argparse
import os

import numpy as np
import pydensecrf.densecrf as dcrf
import scipy.misc as m
from PIL import Image
from tqdm import tqdm

from util import mkdir_if_not_exist


def dense_crf(probs, img=None, n_iters=10,
              sxy_gaussian=(1, 1), compat_gaussian=4,
              kernel_gaussian=dcrf.DIAG_KERNEL,
              normalisation_gaussian=dcrf.NORMALIZE_SYMMETRIC,
              sxy_bilateral=(49, 49), compat_bilateral=5,
              srgb_bilateral=(13, 13, 13),
              kernel_bilateral=dcrf.DIAG_KERNEL,
              normalisation_bilateral=dcrf.NORMALIZE_SYMMETRIC):
    """DenseCRF over unnormalised predictions.
       More details on the arguments at https://github.com/lucasb-eyer/pydensecrf.

    Args:
      probs: class probabilities per pixel.
      img: if given, the pairwise bilateral potential on raw RGB values will be computed.
      n_iters: number of iterations of MAP inference.
      sxy_gaussian: standard deviations for the location component of the colour-independent term.
      compat_gaussian: label compatibilities for the colour-independent term (can be a number, a 1D array, or a 2D array).
      kernel_gaussian: kernel precision matrix for the colour-independent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_gaussian: normalisation for the colour-independent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).
      sxy_bilateral: standard deviations for the location component of the colour-dependent term.
      compat_bilateral: label compatibilities for the colour-dependent term (can be a number, a 1D array, or a 2D array).
      srgb_bilateral: standard deviations for the colour component of the colour-dependent term.
      kernel_bilateral: kernel precision matrix for the colour-dependent term (can take values CONST_KERNEL, DIAG_KERNEL, or FULL_KERNEL).
      normalisation_bilateral: normalisation for the colour-dependent term (possible values are NO_NORMALIZATION, NORMALIZE_BEFORE, NORMALIZE_AFTER, NORMALIZE_SYMMETRIC).

    Returns:
      Refined predictions after MAP inference.
    """
    #     _, h, w, _ = probs.shape
    _, h, w, n_classes = probs.shape

    probs = probs[0].transpose(2, 0, 1).copy(order='C')  # Need a contiguous array.

    d = dcrf.DenseCRF2D(w, h, n_classes)  # Define DenseCRF model.
    U = -np.log(probs)  # Unary potential.x
    U = U.reshape((n_classes, -1))  # Needs to be flat.
    d.setUnaryEnergy(U)
    d.addPairwiseGaussian(sxy=sxy_gaussian, compat=compat_gaussian,
                          kernel=kernel_gaussian, normalization=normalisation_gaussian)
    if img is not None:
        assert (img.shape[1:3] == (h, w)), "The image height and width must coincide with dimensions of the logits."
        d.addPairwiseBilateral(sxy=sxy_bilateral, compat=compat_bilateral,
                               kernel=kernel_bilateral, normalization=normalisation_bilateral,
                               srgb=srgb_bilateral, rgbim=img[0])
    Q = d.inference(n_iters)
    preds = np.array(Q, dtype=np.float32).reshape((n_classes, h, w)).transpose(1, 2, 0)
    return np.expand_dims(preds, 0)


def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)  # only difference


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Postprocessing CRF')
    parser.add_argument('prob_indir', type=str,
                        help='input directory that contains npys in which probability tensors exist')
    parser.add_argument('outdir', type=str,
                        help='output directory that contains predicted labels(pngs)')
    parser.add_argument('--prob_outdir', default=None)
    parser.add_argument('--outimg_shape', default=(1280, 720), nargs=2,
                        help="W H")
    parser.add_argument('--raw_img_indir', type=str, default=None,
                        help="input directory that contains raw imgs(valid:'/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs', test:'/data/ugui0/dataset/adaptation/segmentation_test')")


    args = parser.parse_args()

    args.outimg_shape = [int(x) for x in args.outimg_shape]

    mkdir_if_not_exist(args.outdir)


    for one_file in tqdm(os.listdir(args.prob_indir)):
        one_npy_fn = os.path.join(args.prob_indir, one_file)
        outfn = os.path.join(args.outdir, one_file.replace("npy", "png"))

        #     if os.path.exists(outfn):
        #         continue

        one_prob = np.load(one_npy_fn)
        one_prob = softmax(one_prob)
        one_prob = np.transpose(one_prob, [1, 2, 0])
        one_prob = np.expand_dims(one_prob, 0)
        _, h, w, n_class = one_prob.shape

        if args.raw_img_indir:
            one_raw_img_fn = os.path.join(args.raw_img_indir, one_file.replace("npy", "png"))
            rgb_img = m.imread(one_raw_img_fn)
            rgb_img = Image.fromarray(np.uint8(rgb_img))
            rgb_img = rgb_img.resize((w, h), Image.NEAREST)
            np_rgb_img = np.array(rgb_img)
            np_rgb_img = np.expand_dims(np_rgb_img, 0)
            out = dense_crf(one_prob, img=np_rgb_img)
        else:
            out = dense_crf(one_prob)

        after_crf = np.argmax(out[0, :, :, :19], 2)

        # save prob after crf
        if args.prob_outdir:
            mkdir_if_not_exist(args.prob_outdir)
            out_npy_file = os.path.join(args.prob_outdir, one_file)
            np.save(out_npy_file, np.transpose(out[0], (2, 0, 1)))

        after_crf = Image.fromarray(np.uint8(after_crf))
        after_crf = after_crf.resize(args.outimg_shape, Image.NEAREST)

        after_crf.save(outfn)
