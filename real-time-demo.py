import argparse
import cv2
import torch
import numpy as np
from PIL import Image
from libs.models import encoder4
from libs.models import decoder4
from libs.Matrix import MulLayer
import torchvision.transforms as transforms

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument("--latent", type=int, default=256, help='length of latent vector')
parser.add_argument("--vgg_dir", default='models/vgg_r41.pth', help='pre-trained encoder path')
parser.add_argument("--decoder_dir", default='models/dec_r41.pth', help='pre-trained decoder path')
parser.add_argument("--matrixPath", default='models/matrix_r41_new.pth', help='pre-trained model path')
parser.add_argument("--style_image", type=str, default="Test/style/picasso_self_portrait.jpg", help="path to style image")
parser.add_argument("--record", type=int, default=0, help="set it to 1 for recording into video file")
parser.add_argument("--demo-size", type=int, default=480, help="demo window height, default 480")
opt = parser.parse_args()



# Run the app
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

def run_demo(args, ref, sF, mirror=False):

    # Define the codec and create VideoWriter object
    height = args.demo_size
    width = int(4.0/3*args.demo_size)
    swidth = int(width/4)
    sheight = int(height/4)

    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)
    key = 0
    idx = 0
    while True:
        # read frame
        idx += 1
        ret_val, img = cam.read()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if mirror:
            img = cv2.flip(img, 1)
        cimg = img.copy()
        img = np.array(img).transpose(2, 0, 1) / 255.0

        img = torch.from_numpy(img).unsqueeze(0).float().to(device)

        cF = vgg(img)
        feature, _, _ = matrix(cF['r41'], sF)
        # feature = adaptive_instance_normalization(cF[opt.layer], sF[opt.layer])
        simg = dec(feature) * 255.0
        simg = simg.cpu().clamp(0, 255).data[0].numpy()

        # img = img.cpu().clamp(0, 255).data[0].numpy() * 255.0

        # img = img.transpose(1, 2, 0).astype('uint8')
        simg = simg.transpose(1, 2, 0).astype('uint8')

        # display
        ref = cv2.resize(ref,(swidth, sheight), interpolation = cv2.INTER_CUBIC)
        cimg[0:sheight,0:swidth,:]=ref
        simg = cv2.cvtColor(simg, cv2.COLOR_RGB2BGR)
        cimg = cv2.cvtColor(cimg, cv2.COLOR_RGB2BGR)
        img = np.concatenate((cimg,simg),axis=1)
        cv2.imshow('MSG Demo', img)
        #cv2.imwrite('stylized/%i.jpg'%idx,img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()

    cv2.destroyAllWindows()


transform = transforms.Compose([
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)
style_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(), # range [0, 255] -> [0.0,1.0]
    ]
)

def main():
    vgg.eval()
    matrix.eval()
    dec.eval()
    ref = Image.open(opt.style_image).convert('RGB')
    ref = style_transform(ref).unsqueeze(0).to(device)
    with torch.no_grad():
        sF = vgg(ref)
    # run demo
    ref = ref * 255
    ref = ref.cpu().clamp(0, 255).data[0].numpy()
    ref = ref.transpose(1, 2, 0).astype('uint8')
    run_demo(opt, ref, sF['r41'], mirror=True)

if __name__ == '__main__':
    main()