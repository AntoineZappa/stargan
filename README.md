<p align="center"><img width="40%" src="logo.jpg" /></p>

--------------------------------------------------------------------------------
This repository provides a modification of the PyTorch implementation of [StarGAN](https://arxiv.org/abs/1711.09020) prepared to easily run the network with new datasets. The original repository can be found [here](https://github.com/yunjey/stargan).


## Usage
For running the original experiment with the CelebA dataset:

_NOTE:_ the default parameters on [main.py](main.py) are for running the CelebA experiment as in the original repository, except for `--crop_size` and `--image_resize` that have been set to `None` as default.
```
python main.py --exp_name celeba --image_dir $PATH_TO_CELEBA --crop_size 178 --image_resize 128
```
Check [celeba.py](data_loaders/celeba.py) to perform a translation over a single dataset with different attributes.

<br/><br/>

For running a translation between MNIST and MNIST-M datasets:
```
python main.py --dataset mnist2mnistm --exp_name mnist2mnistm --image_dir $PATH_CONTAINING_BOTH_DATASETS --c_dim 2 --d_conv_dim 32 --g_conv_dim 32 --g_repeat_num 0 --d_repeat_num 4
```
Check [mnist2mnistm.py](data_loaders/mnist2mnistm.py) to perform a translation over different datasets.

## Original Paper
[StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation](https://arxiv.org/abs/1711.09020) <br/>
[Yunjey Choi](https://github.com/yunjey)<sup> 1,2</sup>, [Minje Choi](https://github.com/mjc92)<sup> 1,2</sup>, [Munyoung Kim](https://www.facebook.com/munyoung.kim.1291)<sup> 2,3</sup>, [Jung-Woo Ha](https://www.facebook.com/jungwoo.ha.921)<sup> 2</sup>, [Sung Kim](https://www.cse.ust.hk/~hunkim/)<sup> 2,4</sup>, and [Jaegul Choo](https://sites.google.com/site/jaegulchoo/)<sup> 1,2</sup>    <br/>
<sup>1 </sup>Korea University, <sup>2 </sup>Clova AI Research (NAVER Corp.), <sup>3 </sup>The College of New Jersey, <sup> 4 </sup>HKUST  <br/>
IEEE Conference on Computer Vision and Pattern Recognition ([CVPR](http://cvpr2018.thecvf.com/)), 2018 (<b>Oral</b>) 


## Dependencies
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)

## Citation
If this work is useful for your research, please cite the [paper](https://arxiv.org/abs/1711.09020):
```
@InProceedings{StarGAN2018,
author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2018}
}
```