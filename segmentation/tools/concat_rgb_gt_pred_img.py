"""
Compare predicted visualized png.

Create merged png image that is randomly selected with original RGB image and GT.
"""

import argparse
import os
import random

import numpy as np
from PIL import Image

from util import mkdir_if_not_exist

VIS_GT_DIR_DIC = {
    "city": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_vis_gt/val",
    "city16": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes16_vis_gt/val"
}
RGB_IMG_DIR_DIC = {
    "city": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs",
    "city16": "/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_val_imgs"
}

parser = argparse.ArgumentParser(description='Visualize Some Results')
parser.add_argument('dataset', choices=["gta", "city", "test", "ir", "city16"])
parser.add_argument('--n_img', type=int, default=5)
parser.add_argument('--pred_vis_dirs', type=str, nargs='+',
                    help='result directory that visualized pngs')
parser.add_argument('--outdir', type=str, default="vis_comparison")
parser.add_argument("--rand_sample", action="store_true",
                    help='whether you sample results randomly')

args = parser.parse_args()

rgb_dir = RGB_IMG_DIR_DIC[args.dataset]
vis_gt_dir = VIS_GT_DIR_DIC[args.dataset]

if args.rand_sample:
    rgbfn_list = os.listdir(rgb_dir)
else:
    pickup_id_list = [
        "lindau_000006_000019",
        "frankfurt_000001_021406",
        "frankfurt_000001_041074",
        "frankfurt_000001_002512",
        "frankfurt_000000_009688",
        "frankfurt_000001_040575",
        "munster_000050_000019"
    ]
    rgbfn_list = [x + "_leftImg8bit.png" for x in pickup_id_list]

pickup_rgbfn_list = random.sample(rgbfn_list, args.n_img)
print ("pickup filename list")
print (pickup_rgbfn_list)

all_img_list = []
for rgbfn in pickup_rgbfn_list:
    full_rgbfn = os.path.join(rgb_dir, rgbfn)

    gtfn = rgbfn.replace("leftImg8bit", "gtFine_gtlabels")
    full_gtfn = os.path.join(vis_gt_dir, gtfn)

    one_column_img_list = []
    one_column_img_list.append(Image.open(full_rgbfn))

    one_column_img_list.append(Image.open(full_gtfn))

    for pred_vis_dir in args.pred_vis_dirs:
        full_predfn = os.path.join(pred_vis_dir, rgbfn)
        one_column_img_list.append(Image.open(full_predfn))

    all_img_list.append(one_column_img_list)


def concat_imgs(imgs):
    n_row = len(imgs[0])
    n_col = len(imgs)
    w, h = imgs[0][0].size

    merged_img = Image.new('RGB', (w * n_col, h * n_row))
    for col in range(n_col):
        for row in range(n_row):
            merged_img.paste(imgs[col][row], (w * col, h * row))

    return merged_img


res = concat_imgs(all_img_list)
size = np.array(res.size)
res = res.resize(size / 8)

mkdir_if_not_exist(args.outdir)
shortened_pickup_rgbfn_list = [x.replace("_leftImg8bit.png", "") for x in pickup_rgbfn_list]
pickup_str = "-".join(shortened_pickup_rgbfn_list) + ".pdf"
outfn = os.path.join(args.outdir, pickup_str)
res.save(outfn)
print ("Successfully saved result to %s" % outfn)
