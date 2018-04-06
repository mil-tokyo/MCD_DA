## Maximum Classifier Discrepancy for Domain Adaptation
This is the implementation of Maximum Classifier Discrepancy for Digits Classification in Pytorch.
The code is written by Kuniaki Saito. The work was accepted by CVPR 2018 Oral.

## Getting Started
### Installation
- Install PyTorch (Works on Version 0.2.0_3) and dependencies from http://pytorch.org.
- Install Torch vision from the source.
- Install torchnet as follows
```
pip install git+https://github.com/pytorch/tnt.git@master
```
## Download Dataset
Download MNIST Dataset [here](https://drive.google.com/file/d/1cZ4vSIS-IKoyKWPfcgxFMugw0LtMiqPf/view?usp=sharing). Resized image dataset is contained in the file.
Place it in the directory ./data.
All other datasets should be placed in the directory too.
We will add links to the datasets too.


### Train
For example, if you run an experiment on adaptation from svhn to mnist,
```
python main.py --source svhn --target mnist --num_k 3
```
, where num_k indicates the number of update for generator.
If you want to run an experiment using gradient reversal layer, simply add option --one_step when running this code.
```
python main.py --source svhn --target mnist --one_step
```
