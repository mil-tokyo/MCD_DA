import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm
from util import mkdir_if_not_exist

parser = argparse.ArgumentParser(description='GT Coloring')
parser.add_argument('dataset', choices=["gta", "city", "test", "ir", "city16"])
parser.add_argument('--gt_dir', type=str,
                    default='/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_gt/val')
parser.add_argument('--vis_outdir', type=str,
                    default='/data/unagi0/watanabe/DomainAdaptation/Segmentation/VisDA2017/cityscapes_vis_gt/val')

args = parser.parse_args()

if args.dataset in ["city16", "synthia"]:
    info_json_fn = "./dataset/synthia2cityscapes_info.json"
else:
    info_json_fn = "./dataset/city_info.json"

    # Save visualized predicted pixel labels(pngs)
with open(info_json_fn) as f:
    info_dic = json.load(f)

gtfn_list = os.listdir(args.gt_dir)

for gtfn in tqdm(gtfn_list):
    full_gtfn = os.path.join(args.gt_dir, gtfn)
    img = Image.open(full_gtfn)
    palette = np.array(info_dic['palette'], dtype=np.uint8)
    img.putpalette(palette.flatten())
    mkdir_if_not_exist(args.vis_outdir)
    vis_fn = os.path.join(args.vis_outdir, gtfn)
    img.save(vis_fn)
