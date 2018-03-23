import os
import random

gta_dir = "/data/ugui0/dataset/adaptation/taskcv-2017-public/segmentation/data/"

all_imglist_fn = os.path.join(gta_dir, "images.txt")
test_imglist_fn = os.path.join(gta_dir, "test.txt")
seed = 42
n_test_set = 500

with open(all_imglist_fn) as f:
    img_list = f.readlines()

test_img_list = random.sample(img_list, n_test_set)

print (len(test_img_list))
with open(test_imglist_fn, 'w') as f:
    f.writelines(test_img_list)
