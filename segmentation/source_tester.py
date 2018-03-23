import argparse
import json
import os
from pprint import pprint
import numpy as np
import torch
from PIL import Image
from torch.autograd import Variable
from torch.utils import data
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

from argmyparse import add_additional_params_to_args, fix_img_shape_args
from datasets import get_dataset
from models.model_util import get_full_model, get_optimizer
from transform import Scale, ReLabel, ToLabel
from util import mkdir_if_not_exist, save_dic_to_json, check_if_done

parser = argparse.ArgumentParser(description='Adapt tester for validation data')
parser.add_argument('tgt_dataset', type=str, choices=["gta", "city", "test", "ir", "city16", "synthia", "2d3d"])
parser.add_argument('--split', type=str, default='val', help="'val' or 'test')  is used")
parser.add_argument('trained_checkpoint', type=str, metavar="PTH")
parser.add_argument('--outdir', type=str, default="test_output",
                    help='output directory')
parser.add_argument('--test_img_shape', default=(2048, 1024), nargs=2,
                    help="W H, FOR Valid(2048, 1024) Test(1280, 720)")
parser.add_argument("---saves_prob", action="store_true",
                    help='whether you save probability tensors')

args = parser.parse_args()
args = add_additional_params_to_args(args)
args = fix_img_shape_args(args)

if not os.path.exists(args.trained_checkpoint):
    raise OSError("%s does not exist!" % args.resume)

checkpoint = torch.load(args.trained_checkpoint)
train_args = checkpoint['args']  # Load args!
model = get_full_model(train_args.net, train_args.res, train_args.n_class, train_args.input_ch)
model.load_state_dict(checkpoint['state_dict'])
print ("----- train args ------")
pprint(checkpoint["args"].__dict__, indent=4)
print ("-" * 50)
args.train_img_shape = checkpoint["args"].train_img_shape
print("=> loaded checkpoint '{}'".format(args.trained_checkpoint))

indir, infn = os.path.split(args.trained_checkpoint)

trained_mode = indir.split(os.path.sep)[-2]
args.mode = "%s---%s-%s" % (trained_mode, args.tgt_dataset, args.split)
model_name = infn.replace(".pth", "")

base_outdir = os.path.join(args.outdir, args.mode, model_name)
mkdir_if_not_exist(base_outdir)

json_fn = os.path.join(base_outdir, "param.json")
check_if_done(json_fn)
args.machine = os.uname()[1]
save_dic_to_json(args.__dict__, json_fn)

train_img_shape = tuple([int(x) for x in args.train_img_shape])
test_img_shape = tuple([int(x) for x in args.test_img_shape])

img_transform = Compose([
    Scale(train_img_shape, Image.BILINEAR),
    ToTensor(),
    Normalize([.485, .456, .406], [.229, .224, .225]),

])
label_transform = Compose([
    Scale(train_img_shape, Image.NEAREST),
    ToLabel(),
    ReLabel(255, train_args.n_class - 1),
])

tgt_dataset = get_dataset(dataset_name=args.tgt_dataset, split=args.split, img_transform=img_transform,
                          label_transform=label_transform, test=True, input_ch=train_args.input_ch)

target_loader = data.DataLoader(tgt_dataset, batch_size=1, pin_memory=True)

if torch.cuda.is_available():
    model.cuda()

model.eval()
for index, (imgs, labels, paths) in tqdm(enumerate(target_loader)):
    path = paths[0]
    imgs = Variable(imgs)
    if torch.cuda.is_available():
        imgs = imgs.cuda()

    preds = model(imgs)

    if train_args.net == "psp":
        preds = preds[0]

    if args.saves_prob:
        # Save probability tensors
        prob_outdir = os.path.join(base_outdir, "prob")
        mkdir_if_not_exist(prob_outdir)
        prob_outfn = os.path.join(prob_outdir, path.split('/')[-1].replace('png', 'npy'))
        np.save(prob_outfn, preds[0].data.cpu().numpy())

    # Save predicted pixel labels(pngs)
    if train_args.add_bg_loss:
        pred = preds[0, :train_args.n_class].data.max(0)[1].cpu()
    else:
        pred = preds[0, :train_args.n_class - 1].data.max(0)[1].cpu()

    img = Image.fromarray(np.uint8(pred.numpy()))
    img = img.resize(test_img_shape, Image.NEAREST)
    label_outdir = os.path.join(base_outdir, "label")
    if index == 0:
        print ("pred label dir: %s" % label_outdir)
    mkdir_if_not_exist(label_outdir)
    label_fn = os.path.join(label_outdir, path.split('/')[-1])
    img.save(label_fn)

    #  Save visualized predicted pixel labels(pngs)
    if args.tgt_dataset in ["city16", "synthia"]:
        info_json_fn = "./dataset/synthia2cityscapes_info.json"
    else:
        info_json_fn = "./dataset/city_info.json"

    # Save visualized predicted pixel labels(pngs)
    with open(info_json_fn) as f:
        city_info_dic = json.load(f)

    palette = np.array(city_info_dic['palette'], dtype=np.uint8)
    img.putpalette(palette.flatten())
    vis_outdir = os.path.join(base_outdir, "vis")
    mkdir_if_not_exist(vis_outdir)
    vis_fn = os.path.join(vis_outdir, path.split('/')[-1])
    img.save(vis_fn)
