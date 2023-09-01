# Introduction

## Abstract:
> Image dehazing is a fundamental task for the computer vision and multimedia and usually in the face of the challenge from two aspects, i) the uneven distribution of arbitrary haze and ii) the distortion of image pixels caused by the hazed image. In this paper, we propose an end-to-end trainable framework, named Weighted-Fusion Network with Poly-Scale Convolution (WFN-PSC), to address these dehazing issues. The proposed method is designed based on the Poly-Scale Convolution (PSConv). It can extract the image feature from different scales without upsampling and downsampled, which avoids the image distortion. Beyond this, we design the spatial and channel weighted-fusion modules to make the WFN-PSC model focus on the hard dehazing parts of image from two dimensions. Specifically, we design three Part Architectures followed by the channel weighted-fusion module. Each Part Architecture consists of three PSConv residual blocks and a spatial weighted-fusion module. The experiments on the benchmark demonstrate the dehazing effectiveness of the proposed method. Furthermore, considering that image dehazing is a low-level task in the computer vision, we evaluate the dehazed image on the object detection task and the results show that the proposed method can be a good pre-processing to assist the high-level computer vision task.

## Conda env:
```
name: slx_pytorch_1
channels:
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - http://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - defaults
dependencies:
  - _libgcc_mutex=0.1=main
  - ca-certificates=2021.10.26=h06a4308_2
  - certifi=2021.5.30=py36h06a4308_0
  - ld_impl_linux-64=2.35.1=h7274673_9
  - libffi=3.3=he6710b0_2
  - libgcc-ng=9.1.0=hdf63c60_0
  - libstdcxx-ng=9.1.0=hdf63c60_0
  - ncurses=6.2=he6710b0_1
  - openssl=1.1.1l=h7f8727e_0
  - pip=21.2.2=py36h06a4308_0
  - python=3.6.13=h12debd9_1
  - readline=8.1=h27cfd23_0
  - setuptools=58.0.4=py36h06a4308_0
  - sqlite=3.36.0=hc218d9a_0
  - tk=8.6.11=h1ccaba5_0
  - wheel=0.37.0=pyhd3eb1b0_1
  - xz=5.2.5=h7b6447c_0
  - zlib=1.2.11=h7b6447c_3
prefix: /home/anaconda3/envs/slx_pytorch_1
```

## Dataset url:
*RESIDE-standard* 	
[ITS 百度云](http://tinyurl.com/yaohd3yv),验证码：g0s6

[SOTS 百度云](https://pan.baidu.com/share/init?surl=SSVzR058DX5ar5WL5oBTLg),验证码：s6tu

[HSTS 百度云](https://pan.baidu.com/s/1cl1exWnaFXe3T5-Hr7TJIg),验证码：vzeq

*RESIDE-$\beta$*

[OTS](https://pan.baidu.com/s/1YMYUp5P6FpX_5b7emjgrvA)Passward:  w54h

[RTTS](https://pan.baidu.com/s/1A0MMAnlWmuJ0dXhsbXk4Gg)Passward:  4mv7

## Running:
***Training:***

> 下载好数据集以后，将以下命令的大写部分替换为对应路径
> 
> python train.py --orig_images_path ORIG_IMAGES_PATH --hazy_images_path HAZY_IMAGES_PATH --test_orig_images_path TEST_ORIG_IMAGES_PATH --test_hazy_images_path TEST_HAZY_IMAGES_PATH --lr 0.0001 --num_epochs 100 --train_batch_size 64 --num_workers 4 --display_iter 20 --pretrained PRETRAINED_PATH

***Test:***
> 下载好数据集以后，将以下命令的大写部分替换为对应路径
> 
> python test.py --orig_images_path ORIG_IMAGES_PATH --hazy_images_path HAZY_IMAGES_PATH --test_batch_size 64
>
