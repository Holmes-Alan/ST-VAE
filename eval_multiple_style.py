from __future__ import print_function
import argparse

import os
import torch
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from PIL import Image, ImageOps
from os import listdir
import os
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer_4x

# Training settings
parser = argparse.ArgumentParser(description='LT-VAE multiple style transfer')
parser.add_argument('--image_dataset', type=str, default='Test')
parser.add_argument("--latent", type=int, default=256, help='length of latent vector')
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/matrix_r41_new.pth', help='pre-trained model path')

opt = parser.parse_args()

print(opt)

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


vgg = encoder4()
dec = decoder4()
matrix = MulLayer_4x(z_dim=opt.latent)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

vgg.to(device)
dec.to(device)
matrix.to(device)


def eval(a, b):

    matrix.eval()
    vgg.eval()
    dec.eval()

    content_path = os.path.join(opt.image_dataset, 'content')
    output_path = os.path.join(opt.image_dataset, 'result')
    ref_path = os.path.join(opt.image_dataset, 'style')

    style_tf = test_transform(size=256, crop=True)
    ref_list = []
    for ref_file in os.listdir(ref_path):
        ref = Image.open(os.path.join(ref_path, ref_file)).convert('RGB')

        ref_list.append(style_tf(ref).unsqueeze(0).to(device))

    for cont_file in os.listdir(content_path):
        content = Image.open(os.path.join(content_path, cont_file)).convert('RGB')
        content = transform(content).unsqueeze(0).to(device)
        aa = a/5.0
        bb = b/5.0
        k = (1-aa)*(1 - bb)
        l = bb*(1-aa)
        m = aa*(1-bb)
        n = aa*bb
        with torch.no_grad():
            prediction = chop_forward(k, l, m, n, content, ref_list)

        prediction = prediction * 255.0
        prediction = prediction.clamp(0, 255)

        name = os.path.join(output_path + '/ms_'+str(a)+'_'+str(b)+'.png')
        print(name)
        Image.fromarray(np.uint8(prediction)).save(name)



def test_transform(size, crop):
    transform_list = []
    if size != 0:
        transform_list.append(transforms.Scale(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform =transforms.Compose(transform_list)

    return transform


transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)


def chop_forward(k, l, m, n, content, ref):

    with torch.no_grad():
        sF_1 = vgg(ref[0])
        sF_2 = vgg(ref[1])
        sF_3 = vgg(ref[2])
        sF_4 = vgg(ref[3])
        cF = vgg(content)
        feature, _ = matrix(k, l, m, n, cF[opt.layer], sF_1[opt.layer], sF_2[opt.layer], sF_3[opt.layer], sF_4[opt.layer])
        transfer = dec(feature)

        transfer = transfer.data[0].cpu().permute(1, 2, 0)

    return transfer




##Eval Start!!!!
for a in range(0, 5, 1):
    for b in range(0, 5, 1):
        eval(a, b)
