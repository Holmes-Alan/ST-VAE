from __future__ import print_function
import argparse

import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from os.path import join
import time
from PIL import Image, ImageOps
import os
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer

# Training settings
parser = argparse.ArgumentParser(description='LT-VAE Style transfer')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
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
matrix = MulLayer(z_dim=opt.latent)
vgg.load_state_dict(torch.load(opt.vgg_dir))
dec.load_state_dict(torch.load(opt.decoder_dir))
matrix.load_state_dict(torch.load(opt.matrixPath))

vgg.to(device)
dec.to(device)
matrix.to(device)
print('===> Loading datasets')


def eval():

    matrix.eval()
    vgg.eval()
    dec.eval()
    content_path = os.path.join(opt.image_dataset, 'content')
    output_path = os.path.join(opt.image_dataset, 'result')
    ref_path = os.path.join(opt.image_dataset, 'style')

    for cont_file in os.listdir(content_path):
        for ref_file in os.listdir(ref_path):
            t0 = time.time()
            content = Image.open(os.path.join(content_path, cont_file)).convert('RGB')
            ref = Image.open(os.path.join(ref_path, ref_file)).convert('RGB')

            content = transform(content).unsqueeze(0).to(device)
            ref = transform(ref).unsqueeze(0).to(device)

            with torch.no_grad():
                sF = vgg(ref)
                cF = vgg(content)
                feature, _, _ = matrix(cF['r41'], sF['r41'])
                prediction = dec(feature)

                prediction = prediction.data[0].cpu().permute(1, 2, 0)

            t1 = time.time()
            #print("===> Processing: %s || Timer: %.4f sec." % (str(i), (t1 - t0)))

            prediction = prediction * 255.0
            prediction = prediction.clamp(0, 255)

            file_name = cont_file.split('.')[0] + '_' + ref_file.split('.')[0] + '.jpg'
            save_name = os.path.join(output_path, file_name)
            Image.fromarray(np.uint8(prediction)).save(save_name)



transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)



##Eval Start!!!!
eval()
