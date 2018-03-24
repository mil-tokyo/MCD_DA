
![](overview.png)
<br>

**This is the project page for Maximum Classifier Discrepancy.
The work was accepted by CVPR 2018 Oral.**
[[Paper Link(arxiv)]](https://arxiv.org/abs/1712.02560).
<br>

## Abstract
We propose a new approach for unsupervised domain adaptation, which attempts to align distributions of source and target by utilizing the task-specific decision boundaries.
We propose to maximize the discrepancy between two classifiers' outputs to detect target samples that are far from the support of the source. A feature generator learns to generate target features near the support to minimize the discrepancy.
Our method is applicable to classification and semantic segmentation. The implementation is availble now !

## Example
Our method demonstrates good performance both on  classification and semantic segmentation for unsupervised domain adaptation.
The examples of semantic segmentation are shown here.

<br>
![](result_seg.png)
<br>

## Codes
[[Classification]](https://github.com/mil-tokyo/MCD_DA/tree/master/classification) [[Segmentation]](https://github.com/mil-tokyo/MCD_DA/tree/master/segmentation)

## Citation
If you use this code for your research, please cite our papers (This will be updated when cvpr paper is publicized).
```
@article{saito2017maximum,
  title={Maximum Classifier Discrepancy for Unsupervised Domain Adaptation},
  author={Saito, Kuniaki and Watanabe, Kohei and Ushiku, Yoshitaka and Harada, Tatsuya},
  journal={arXiv preprint arXiv:1712.02560},
  year={2017}
}
```
