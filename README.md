# ST-VAE
Multiple style transfer via variational autoencoder

By Zhi-Song Liu, Vicky Kalogeiton and Marie-Paule Cani

This repo only provides simple testing codes, pretrained models and the network strategy demo.

We propose a Multiple style transfer via variational autoencoder (ST-VAE)

Please check our [paper](http://www.lix.polytechnique.fr/~kalogeiton/2021/icip-style-transfer/)
or arxiv [paper](https://arxiv.org/abs/2110.07375)

# BibTex

        @InProceedings{Liu2021stvae,
            author = {Zhi-Song Liu and Wan-Chi Siu and Marie-Paule Cani},
            title = {Multiple Style Transfer via Variational AutoEncoder},
            booktitle = {2021 IEEE International Conference on Image Processing(ICIP)},
            month = {Oct},
            year = {2021}
        }
        
## For proposed ST-VAE model, we claim the following points:

• First working on using Variational AutoEncoder for image style transfer.

• Multiple style transfer by proposed VAE based Linear Transformation.

# Dependencies
    Python > 3.0
    Pytorch > 1.0
    NVIDIA GPU + CUDA

# Complete Architecture
The complete architecture is shown as follows,

![network](/figure/figure1.PNG)

# Visualization
## 1. Single style transfer

![st_single](/figure/figure3.PNG)

## 2. Multiple style transfer

![st_multiple](/figure/figure2.PNG)

# Implementation
## 1. Quick testing
---------------------------------------
1. Download pre-trained models from

https://drive.google.com/file/d/1WZrvjCGBO1mpggkdJiaw8jp-6ywbXn4J/view?usp=sharing

and copy them to the folder "models"

2. Put your content image under "Test/content" and your style image under "Test/style"

3. For single style transfer, run 
```sh
$ python eval.py 
```
The stylized images will be in folder "Test/result"
4. For multiple style transfer, run
```sh
$ python eval_multiple_style.py
```
5. For real-time demo, run
```sh
$ python real-time-demo.py --style_image Test/style/picasso_self_portrait.jpg
```
6. For training, put the training images under the folder "train_data"

download MS-COCO dataset from https://cocodataset.org/#home and put it under "train_data/content"
download Wikiart from https://www.wikiart.org/ and put them under "train_data/style"
then run,
 ```sh
$ python train.py
```




Special thanks to the contributions of Jakub M. Tomczak for their [LT](https://github.com/sunshineatnoon/LinearStyleTransfer) on their LT computation
