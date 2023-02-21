# SimVP: Simpler yet Better Video Prediction
![GitHub stars](https://img.shields.io/github/stars/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction)  ![GitHub forks](https://img.shields.io/github/forks/gaozhangyang/SimVP-Simpler-yet-Better-Video-Prediction?color=green) 

**In the example, the default epoch is 50. Please read our paper, and train 1000~2000 epochs for repruducing this work!** I will not respond to such a lowly mistake.

**The pre-trained models and benchmarks will be available in SimVPv2.**

**SimVPv2 is available on https://github.com/chengtan9907/SimVPv2, which performs better than SimVP (15.05 MSE on Moving MNIST) and is in the review process.** If our work is helpful for your research, we would hope you give us a star and citation. Thanks!

This repository contains the implementation code for paper:

**SimVP: Simpler yet Better Video Prediction**  
[Zhangyang Gao](https://westlake-drug-discovery.github.io/zhangyang_gao.html), [Cheng Tan](https://westlake-drug-discovery.github.io/cheng_tan.html), [Lirong Wu](https://lirongwu.github.io/), [Stan Z. Li](https://scholar.google.com/citations?user=Y-nyLGIAAAAJ&hl). In [CVPR](), 2022.
## Introduction

<p align="center">
    <img src="./readme_figures/overall_framework.png" width="600"> <br>
</p>

From CNN, RNN, to ViT, we have witnessed remarkable advancements in video prediction, incorporating auxiliary inputs, elaborate neural architectures, and sophisticated training strategies. We admire these progresses but are confused about the necessity: is there a simple method that can perform comparably well? This paper proposes SimVP, a simple video prediction model that is completely built upon CNN and trained by MSE loss in an end-to-end fashion. Without introducing any additional tricks and complicated strategies, we can achieve state-of-the-art performance on five benchmark datasets. Through extended experiments, we demonstrate that SimVP has strong generalization and extensibility on real-world datasets. The significant reduction of training cost makes it easier to scale to complex scenarios. We believe SimVP can serve as a solid baseline to stimulate the further development of video prediction.

## Dependencies
* torch
* scikit-image=0.16.2
* numpy
* argparse
* tqdm

## Overview

* `API/` contains dataloaders and metrics.
* `main.py` is the executable python file with possible arguments.
* `model.py` contains the SimVP model.
* `exp.py` is the core file for training, validating, and testing pipelines.

## Install

This project has provided an environment setting file of conda, users can easily reproduce the environment by the following commands:
```
  conda env create -f environment.yml
  conda activate SimVP
```

### Moving MNIST dataset

```
  cd ./data/moving_mnist
  bash download_mmnist.sh
```

### TaxiBJ dataset

We provide a [Dropbox](https://www.dropbox.com/sh/l9drnyeftcmy3j1/AACCgUyOj2akPNBwFAe9W1-ia?dl=0) to download TaxiBJ dataset. Users can download this dataset and put it into `./data/taxibj`.

### KTH dataset

We provide a [Dropbox](https://www.dropbox.com/sh/8d3uwyp4jru0yih/AABP-nXlN6eHW2xOrkfCn7Woa?dl=0) to download the KTH dataset.


## Citation

If you are interested in our repository and our paper, please cite the following paper:

```
@InProceedings{Gao_2022_CVPR,
    author    = {Gao, Zhangyang and Tan, Cheng and Wu, Lirong and Li, Stan Z.},
    title     = {SimVP: Simpler Yet Better Video Prediction},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {3170-3180}
}
```

## Contact

If you have any questions, feel free to contact us through email (tancheng@westlake.edu.cn, gaozhangyang@westlake.edu.cn). Enjoy!
