# coding: utf-8
import argparse
import os

import matplotlib
from PIL import Image

from transform import Colorize

matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from util import mkdir_if_not_exist

label_list = [
    "road",
    "sidewalk",
    "building",
    "wall",
    "fence",
    "pole",
    "light",
    "sign",
    "vegetation",
    "terrain",
    "sky",
    "person",
    "rider",
    "car",
    "truck",
    "bus",
    "train",
    "motocycle",
    "bicycle",
    "background"
]

values = np.arange(len(label_list))
N_CLASS = len(label_list)


def one_vis_with_legend(indir, outdir):
    for one_file in tqdm(os.listdir(indir)):
        fullpath = os.path.join(indir, one_file)
        hard_to_see_img = m.imread(fullpath)
        im = plt.imshow(hard_to_see_img.astype(np.int64), interpolation='none', cmap="jet", vmin=0, vmax=N_CLASS - 1)
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=label_list[i]) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        outfn = os.path.join(outdir, one_file)
        plt.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        plt.close()


def vis_with_legend(indir_list, raw_rgb_dir, outdir, raw_gray_dir=None, gt_dir=None, ext="png"):
    n_imgs = 1 + len(indir_list)
    if raw_gray_dir:
        n_imgs += 1
    if gt_dir:
        n_imgs += 1

    mkdir_if_not_exist(outdir)

    n_row = 2
    n_col = int(round(float(n_imgs) / n_row))

    # img_fn_list = os.listdir(raw_rgb_dir)
    img_fn_list = os.listdir(indir_list[0])

    for one_img_fn in tqdm(img_fn_list):
        fig = plt.figure()  # sharex=True, sharey=True)
        ax_list = []
        ax_list.append(fig.add_subplot(n_row, n_col, 1))
        raw_img = Image.open(os.path.join(raw_rgb_dir, one_img_fn))

        ax_list[0].imshow(raw_img)
        ax_list[0].axis("off")
        ax_list[0].set_xticklabels([])
        ax_list[0].set_yticklabels([])

        offset = 1

        if raw_gray_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            raw_img = Image.open(os.path.join(raw_gray_dir, one_img_fn))

            ax_list[offset].imshow(raw_img, cmap='gray')
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])
            offset += 1

        if gt_dir:
            ax_list.append(fig.add_subplot(n_row, n_col, offset + 1))
            gt_img = Image.open(os.path.join(gt_dir, one_img_fn.replace("leftImg8bit", "gtFine_gtlabels")))
            ax_list[offset].imshow(gt_img, vmin=0, vmax=N_CLASS - 1, interpolation='none', cmap="jet")
            ax_list[offset].axis("off")
            ax_list[offset].set_xticklabels([])
            ax_list[offset].set_yticklabels([])
            offset += 1

        # ax_list[0].set_aspect('equal')
        for i, indir in enumerate(indir_list):
            # hard_to_see_img = m.imread(os.path.join(indir, one_img_fn))
            hard_to_see_img = Image.open(os.path.join(indir, one_img_fn)).resize(raw_img.size)
            hard_to_see_img = np.array(hard_to_see_img)

            ax_list.append(fig.add_subplot(n_row, n_col, i + offset + 1))
            im = ax_list[i + offset].imshow(hard_to_see_img.astype(np.uint8), vmin=0, vmax=N_CLASS - 1,
                                            interpolation='none',
                                            cmap="jet")
            ax_list[i + offset].axis("off")
            ax_list[i + offset].set_xticklabels([])
            ax_list[i + offset].set_yticklabels([])
            ax_list[i + offset].set_title(indir.replace("outputs/", "").replace("/label", "").replace("/", "\n"),
                                          fontsize=4)
            # ax_list[i + 1].set_aspect('equal')

        fig.subplots_adjust(wspace=0, hspace=0)

        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label=label_list[i]) for i in range(len(values))]
        # lgd = fig.legend(handles=patches, labels=label_list, bbox_to_anchor=(1.05, 1), borderaxespad=0.,
        #                  fontsize=7, loc='upper left')  # loc=2
        if n_col * 2 <= N_CLASS:
            n_legend_col = n_col * 2
        else:
            n_legend_col = N_CLASS
        lgd = plt.legend(patches, label_list, loc='lower center', bbox_to_anchor=(0, 0, 1, 1),
                         bbox_transform=plt.gcf().transFigure, ncol=n_legend_col, fontsize=5)

        # fig.tight_layout()
        outfn = os.path.join(outdir, one_img_fn)
        outfn = os.path.splitext(outfn)[0] + '.%s' % ext

        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0, bbox_extra_artists=(lgd,), dpi=300)
        plt.close()


# TODO This is not work
def vis_using_Colorize(indir_list, outdir):
    indir = indir_list[0]
    # outdir = os.path.join(os.path.split(indir)[0], "vis_labels")
    mkdir_if_not_exist(outdir)

    for one_file in tqdm(os.listdir(indir)):
        fullpath = os.path.join(indir, one_file)
        hard_to_see_img = m.imread(fullpath)
        # outputs = outputs[0, :19].data.max(0)[1]
        # outputs = outputs.view(1, outputs.size()[0], outputs.size()[1])
        outputs = hard_to_see_img  # TODO this should be fixed
        output = Colorize()(outputs)
        output = np.transpose(output.cpu().numpy(), (1, 2, 0))
        img = Image.fromarray(output, "RGB")
        img = img.resize(hard_to_see_img.shape, Image.NEAREST)

        outfn = os.path.join(outdir, one_file)
        plt.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0)
        img.save(outfn)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='visualize labels')
    parser.add_argument('--indir_list', type=str, nargs='*',
                        help='result directory that contains predicted labels(pngs)')
    parser.add_argument('--outdir', type=str, required=True,
                        help='visualized dir')
    parser.add_argument("--raw_rgb_dir", type=str, default="/data/ugui0/dataset/adaptation/segmentation_test",
                        help="raw img dir")
    parser.add_argument("--raw_gray_dir", type=str, default=None,
                        help="raw img dir2")
    parser.add_argument("--gt_dir", type=str, default=None,
                        help="gt dir")
    parser.add_argument("--way", type=str, default="legend", help="legend or colorize",
                        choices=['legend', 'colorize'])
    parser.add_argument("--ext", type=str, default="png")

    args = parser.parse_args()

    if args.way == "legend":
        vis_with_legend(args.indir_list, args.raw_rgb_dir, args.outdir, args.raw_gray_dir, args.gt_dir,
                        args.ext)
    elif args.way == "colorize":  # TODO
        vis_using_Colorize(args.indir_lis, args.outdir)
