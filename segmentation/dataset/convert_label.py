import argparse
import json
import os

import numpy as np
from PIL import Image
from tqdm import tqdm

def swap_labels(np_original_gt_im, class_convert_mat):
    np_processed_gt_im = np.zeros(np_original_gt_im.shape)
    for swap in class_convert_mat:
        ind_swap = np.where(np_original_gt_im == swap[0])
        np_processed_gt_im[ind_swap] = swap[1]
    processed_gt_im = Image.fromarray(np.uint8(np_processed_gt_im))
    return processed_gt_im


def convert_citylabelTo16label():
    with open('./synthia2cityscapes_info.json', 'r') as f:
        paramdic = json.load(f)

    class_ind = paramdic['city2common']

    city_gt_dir = "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/gtFine"
    split_list = ["train", "test", "val"]

    original_suffix = "labelIds"
    processed_suffix = "label16IDs"

    for split in tqdm(split_list):
        base_dir = os.path.join(city_gt_dir, split)
        place_list = os.listdir(base_dir)
        for place in tqdm(place_list):
            target_dir = os.path.join(base_dir, place)
            pngfn_list = os.listdir(target_dir)
            original_pngfn_list = [x for x in pngfn_list if original_suffix in x]

            for pngfn in tqdm(original_pngfn_list):
                gt_fn = os.path.join(target_dir, pngfn)
                original_gt_im = Image.open(gt_fn)

                processed_gt_im = swap_labels(np.array(original_gt_im), class_ind)
                outfn = gt_fn.replace(original_suffix, processed_suffix)
                processed_gt_im.save(outfn, 'PNG')


def convert_synthialabelTo16label():
    with open('./synthia2cityscapes_info.json', 'r') as f:
        paramdic = json.load(f)

    class_ind = np.array(paramdic['synthia2common'])

    synthia_gt_dir = "/data/ugui0/dataset/adaptation/synthia/RAND_CITYSCAPES/GT"

    # original_dir = os.path.join(synthia_gt_dir, "LABELS") # Original dir but this contains strange files
    original_dir = "/data/ugui0/dataset/adaptation/synthia/new_synthia/segmentation_annotation/SYNTHIA/GT/parsed_LABELS"  # Not original. Downloaded from http://crcv.ucf.edu/data/adaptationseg/ICCV_dataset.zip
    processed_dir = os.path.join(synthia_gt_dir, "LABELS16")

    original_pngfn_list = os.listdir(original_dir)

    for pngfn in tqdm(original_pngfn_list):
        gt_fn = os.path.join(original_dir, pngfn)
        original_gt_im = Image.open(gt_fn)
        processed_gt_im = swap_labels(np.array(original_gt_im), class_ind)
        outfn = os.path.join(processed_dir, pngfn)
        processed_gt_im.save(outfn, 'PNG')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Label Ids')
    parser.add_argument('dataset', type=str, choices=["city", "synthia"])
    args = parser.parse_args()
    if args.dataset == "city":
        convert_citylabelTo16label()
    else:
        convert_synthialabelTo16label()
