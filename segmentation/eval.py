from __future__ import print_function

import argparse
import json
import os
from os.path import join
import matplotlib
import numpy as np
import pandas as pd
from PIL import Image

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

import collections
from sklearn.metrics import accuracy_score


def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def save_colorful_images(prediction, filename, palette, postfix='_color.png'):
    im = Image.fromarray(palette[prediction.squeeze()])
    im.save(filename[:-4] + postfix)


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def save_result(outfn, name_classes, mIoUs):
    result_df = pd.DataFrame({
        'class': name_classes,
        'IoU': mIoUs
    })
    ave_df = pd.DataFrame({
        'class': "mIoU",
        'IoU': result_df.IoU.mean(),
    }, index=[result_df.shape[0]])
    result_df = result_df.append(ave_df)
    result_df.set_index("class", inplace=True)
    result_df.to_csv(outfn)
    print('The result is saved at %s !' % outfn)


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    #     plt.text(j, i, format(cm[i, j], fmt),
    #              horizontalalignment="center",
    #              color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


def compute_mIoU(gt_dir, pred_dir, devkit_dir='', dset='cityscapes', calc_confmat=False, get_label_dist=False,
                 add_bg_loss=False, is_label16=False, split='val'):
    """
    Compute IoU given the predicted colorized images and 
    """
    if is_label16:
        with open("./dataset/synthia2cityscapes_info.json", 'r') as fp:
            info = json.load(fp)
    else:
        with open(join(devkit_dir, 'data', dset, 'info.json'), 'r') as fp:
            info = json.load(fp)
    num_classes = np.int(info['classes'])

    if is_label16:
        name_classes = np.array(info['common_label'], dtype=np.str)
        mapping = np.array(info['city2common'], dtype=np.int)
    else:
        name_classes = np.array(info['label'], dtype=np.str)
        mapping = np.array(info['label2train'], dtype=np.int)

    if add_bg_loss:
        num_classes = np.int(info['classes']) + 1
        name_classes = np.array(info['label'] + ["background"], dtype=np.str)

    print(name_classes)
    print("pred path: %s" % os.path.abspath(pred_dir))

    palette = np.array(info['palette'], dtype=np.uint8)
    hist = np.zeros((num_classes, num_classes))
    image_path_list = join(devkit_dir, 'data', dset, 'image.txt')
    label_path_list = join(devkit_dir, 'data', dset, 'label.txt')
    if split == 'test':
        label_path_list = '/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/gtFine/test.txt'
        image_path_list = '/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/leftImg8bit/test.txt'
    gt_imgs = open(label_path_list, 'rb').read().splitlines()

    pred_imgs = open(image_path_list, 'rb').read().splitlines()

    labels_list = []
    preds_list = []

    n_total_pixel = 0
    total_acc = 0
    for ind, gt_fn in tqdm(enumerate(gt_imgs)):
        pred_fn = join(pred_dir,
                       pred_imgs[ind].split('/')[-1].replace('gtFine_labelIds', 'leftImg8bit').replace('label', ""))
        pred = Image.open(pred_fn)

        if is_label16:
            label = Image.open(join(gt_dir, gt_imgs[ind].replace('labelIds', 'label16IDs')))
        else:
            gt_fn = join(gt_dir, gt_imgs[ind].replace('labelIds', 'gtlabels'))
            if split == 'test':
                gt_fn = join(gt_dir, gt_imgs[ind])
            # print(gt_fn)
            label = Image.open(gt_fn)

        pred = pred.resize(label.size)
        pred = np.array(pred)
        label = np.array(label)

        if add_bg_loss:
            bk = np.where(label == 255)

            label[bk] = 19

        not_background_idxes = np.where(label != 255)
        # print(not_background_idxes)
        label = label[not_background_idxes]
        pred = pred[not_background_idxes]

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)
        n_pixel = len(label)
        one_acc = accuracy_score(label, pred) * n_pixel
        total_acc += one_acc
        n_total_pixel += n_pixel

        if get_label_dist or calc_confmat:
            labels_list.append(label.flatten())
            preds_list.append(pred.flatten())

            # if ind > 0 and ind % 100 == 0:
            #     print('{:d} / {:d}: {:0.2f}'.format(ind, len(gt_imgs), 100 * np.mean(per_class_iu(hist))))

    labels_list = np.array(labels_list).flatten()
    preds_list = np.array(preds_list).flatten()

    if get_label_dist:
        label_cnt_ser = pd.Series(dict(collections.Counter(labels_list)))
        pred_cnt_ser = pd.Series(dict(collections.Counter(preds_list)))
        label_distribution_df = pd.concat([label_cnt_ser, pred_cnt_ser], axis=1)
        label_distribution_df.columns = ["true", "pred"]

        outfn = os.path.join(os.path.split(pred_dir)[0].replace("label", ""), "label_distribution.csv")
        label_distribution_df.to_csv(outfn)
        print("label distribution file was saved to %s" % outfn)

    if calc_confmat:
        conf_mat = confusion_matrix(labels_list, preds_list)
        fig = plt.figure()
        plot_confusion_matrix(conf_mat, classes=name_classes, title='Confusion matrix')
        outfn = os.path.join(os.path.split(pred_dir)[0].replace("label", ""), "conf_mat.pdf")
        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)
        outfn = os.path.join(os.path.split(pred_dir)[0].replace("label", ""), "conf_mat.png")
        fig.savefig(outfn, transparent=True, bbox_inches='tight', pad_inches=0, dpi=300)

    total_acc /= n_total_pixel
    total_acc *= 100

    print("pixel acc(without background): %s" % total_acc)

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs) * 100, 2)))

    if add_bg_loss:
        print('===> mIoU without background: ' + str(round(np.nanmean(mIoUs[:19]) * 100, 2)))

    iou_str = ""
    for iou in mIoUs:
        iou_str += str(iou * 100) + ", "
    print(iou_str[:-2])

    outfn = os.path.join(os.path.split(pred_dir)[0].replace("label", ""), "eval_result.csv")
    save_result(outfn, name_classes, mIoUs)

    return mIoUs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('dset', default='city', help='For the challenge use the validation set of cityscapes.',
                        choices=['city', "city16", 'gta'])
    parser.add_argument('pred_dir', type=str, help='directory which stores CityScapes val pred images')
    parser.add_argument('--devkit_dir', default='/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation',
                        help='base directory of taskcv2017/segmentation')
    parser.add_argument('--split', type=str, default='test', help="this only supported in IR dataset")
    parser.add_argument('--calc_confmat', action="store_true",
                        help='whether you calculate confusion matrix or not')
    parser.add_argument('--get_label_dist', action="store_true",
                        help='whether you calculate confusion matrix or not')
    parser.add_argument('--add_bg_loss', action="store_true",
                        help='whether you calculate confusion matrix or not')
    args = parser.parse_args()

    if args.dset in ["city", "city16"]:
        gt_dir = "/data/ugui0/ksaito/D_A/image_citiscape/www.cityscapes-dataset.com/file-handling/gtFine/val"  # +args.split
        is_label16 = True if args.dset == "city16" else False
        compute_mIoU(gt_dir, args.pred_dir, args.devkit_dir, "cityscapes", args.calc_confmat, args.get_label_dist,
                     args.add_bg_loss, is_label16, split=args.split)
    elif args.dset == "gta":
        gt_dir = "/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation/data/"

        compute_mIoU(gt_dir, args.pred_dir, args.devkit_dir, "gta", args.calc_confmat, args.get_label_dist,
                     args.add_bg_loss)
