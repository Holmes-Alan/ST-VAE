import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
from data import get_training_set
from libs.Matrix import MulLayer
import torchvision.utils as vutils
import torch.backends.cudnn as cudnn
from libs.utils import print_options
from libs.Criterion import LossCriterion
from libs.models import encoder4
from libs.models import decoder4
from libs.models import encoder5 as loss_network
from torch.utils.data import DataLoader
from torch.autograd import Variable

parser = argparse.ArgumentParser()
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--loss_network_dir", default='models/vgg_r51.pth', help='used for loss network')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument("--matrix_dir", default='model/matrix_r41_new.pth', help='pre-trained matrix path')
parser.add_argument("--data_dir", default="train_data", help='path to training dataset')
parser.add_argument('--data_augmentation', type=bool, default=True)
parser.add_argument('--threads', type=int, default=6, help='number of threads for data loader to use')
parser.add_argument("--outf", default="output/", help='folder to output images and model checkpoints')
parser.add_argument("--content_layers", default="r41", help='layers for content')
parser.add_argument("--style_layers", default="r11,r21,r31,r41", help='layers for style')
parser.add_argument("--batchSize", type=int,default=8, help='batch size')
parser.add_argument("--lr", type=float, default=1e-4, help='learning rate')
parser.add_argument("--content_weight", type=float, default=1.0, help='content loss weight')
parser.add_argument("--style_weight", type=float, default=0.02, help='style loss weight, 0.02 for origin')
parser.add_argument("--log_interval", type=int, default=500, help='log interval')
parser.add_argument("--gpu_id", type=int, default=0, help='which gpu to use')
parser.add_argument("--save_interval", type=int, default=5000, help='checkpoint save interval')
parser.add_argument("--layer", default="r41", help='which features to transfer, either r31 or r41')
parser.add_argument("--latent", type=int, default=1024, help='length of latent vector')
parser.add_argument('--nEpochs', type=int, default=5000, help='number of epochs to train for')
parser.add_argument('--snapshots', type=int, default=5, help='Snapshots')
parser.add_argument('--start_iter', type=int, default=1, help='Starting Epoch')

################# PREPARATIONS #################
opt = parser.parse_args()
opt.content_layers = opt.content_layers.split(',')
opt.style_layers = opt.style_layers.split(',')
opt.cuda = torch.cuda.is_available()
if(opt.cuda):
    torch.cuda.set_device(opt.gpu_id)

os.makedirs(opt.outf,exist_ok=True)
cudnn.benchmark = True
print_options(opt)

################# DATA #################
print('===> Loading datasets')
train_set = get_training_set(opt.data_dir, opt.data_augmentation)
training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

################# MODEL #################
vgg5 = loss_network()

matrix = MulLayer(z_dim=opt.latent)
vgg = encoder4()
dec = decoder4()

if os.path.exists(opt.vgg_dir):
    vgg.load_state_dict(torch.load(opt.vgg_dir))
    print('enocder model is loaded!')
if os.path.exists(opt.decoder_dir):
    dec.load_state_dict(torch.load(opt.decoder_dir))
    print('decoder model is loaded!')
if os.path.exists(opt.loss_network_dir):
    vgg5.load_state_dict(torch.load(opt.loss_network_dir))
    print('VGG model is loaded!')

if opt.pretrained:
    if os.path.exists(opt.matrix_dir):
        matrix.load_state_dict(torch.load(opt.matrix_dir))
        print('pretrained matrix model is loaded!')

for param in vgg.parameters():
    param.requires_grad = False
for param in vgg5.parameters():
    param.requires_grad = False
for param in dec.parameters():
    param.requires_grad = False

################# LOSS & OPTIMIZER #################
criterion = LossCriterion(opt.style_layers,
                          opt.content_layers,
                          opt.style_weight,
                          opt.content_weight)
optimizer = optim.Adam(matrix.parameters(), opt.lr)

################# GPU  #################
if(opt.cuda):
    vgg.cuda()
    dec.cuda()
    vgg5.cuda()
    matrix.cuda()

################# TRAINING #################

def train(epoch):
    epoch_loss = 0

    for iteration, batch in enumerate(training_data_loader, 1):
        content, target, style = Variable(batch[0]), Variable(batch[1]), Variable(batch[2])
        content = content.cuda()
        target = target.cuda()
        style = style.cuda()

        optimizer.zero_grad()

        # forward
        sF = vgg(style)
        cF = vgg(content)

        # if(opt.layer == 'r41'):
        feature, transmatrix, KL = matrix(cF[opt.layer], sF[opt.layer])
        # else:
        #     feature, transmatrix = matrix(cF, sF)
        transfer = dec(feature)

        sF_loss = vgg5(style)
        cF_loss = vgg5(content)
        tF = vgg5(transfer)
        loss, styleLoss, contentLoss, KL_loss = criterion(tF, sF_loss, cF_loss, KL)

        # backward & optimization
        loss.backward()
        optimizer.step()


        print("===> Epoch[{}]({}/{}): loss: {:.4f} || content: {:.4f} || style: {:.4f} KL: {:.4f}.".format(epoch, iteration,
                                                                                 len(training_data_loader), loss, contentLoss, styleLoss, KL_loss,))

    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))

    return content, style, transfer


for epoch in range(opt.start_iter, opt.nEpochs + 1):
    content, style, transfer = train(epoch)

    # learning rate is decayed by a factor of 10 every half of total epochs
    if (epoch + 1) % 100 == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] /= 10.0
        print('Learning rate decay: lr={}'.format(optimizer.param_groups[0]['lr']))

    if epoch % (opt.snapshots) == 0:
        content = content.clamp(0, 1).cpu().data
        style = style.clamp(0, 1).cpu().data
        transfer = transfer.clamp(0, 1).cpu().data
        concat = torch.cat((content, style, transfer), dim=0)
        vutils.save_image(concat, '%s/%d.png' % (opt.outf, epoch), normalize=True, scale_each=True, nrow=opt.batchSize)

        torch.save(matrix.state_dict(), '%s/%s_epoch_%d.pth' % (opt.outf, opt.layer, epoch))
