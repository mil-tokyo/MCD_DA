"""
Compare predicted visualized png.

Create merged png image.
"""

from PIL import Image
import sys
import os


def merge_four_images(files, outimg):
    """
    Create merged_img.

    files : list of 4 image file paths
    merged_img(PIL Image)
        - topleft: 1st
        - bottomleft: 2nd
        - topright: 3rd
        - bottomright: 4th
    """
    assert len(files) == 4

    img = [Image.open(file_) for file_ in files]

    img_size = img[0].size
    merged_img = Image.new('RGB', (img_size[0]*2, img_size[1]*2))
    for row in range(2):
        for col in range(2):
            merged_img.paste(img[row*2+col], (img_size[0]*row, img_size[1]*col))

    merged_img.save(outimg)


def main(vis_dirs, outdir):
    """Out merged_imgs from 4 directories (one directory is gt directory)."""
    assert len(vis_dirs) == 4

    if not os.path.exists(outdir):
        os.mkdir(outdir)

    for i, filename in enumerate(os.listdir(vis_dirs[0])):
        if i%100 == 0:
            print(i)
        try:
            files = [os.path.join(vis_dir, filename) for vis_dir in vis_dirs]
            outimg = os.path.join(outdir, filename)
            merge_four_images(files, outimg)
        except:
            print(filename)

if __name__ == '__main__':
    args = sys.argv

    """num of args need to be 3."""

    # merge_four_images(args[1:], 'sample_merged.png')
    vis_dirs = ['/data/ugui0/dataset/adaptation/segmentation_test'] + args[1:]

    for i in range(20):
        outdir = 'merged_imgs/merged_imgs_{0}'.format(i)
        if os.path.exists(outdir):
            continue
        else:
            break

    main(vis_dirs, outdir)
