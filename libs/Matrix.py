import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self, matrixSize=32):
        super(CNN,self).__init__()

        self.convs = nn.Sequential(nn.Conv2d(512,256,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(256,128,3,1,1),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128,matrixSize,3,1,1))

        # 32x8x8
        self.fc = nn.Linear(matrixSize*matrixSize,matrixSize*matrixSize)
        #self.fc = nn.Linear(32*64,256*256)

    def forward(self,x):
        out = self.convs(x)
        # 32x8x8
        b,c,h,w = out.size()
        out = out.view(b,c,-1)
        # 32x64
        out = torch.bmm(out,out.transpose(1,2)).div(h*w)
        # 32x32
        out = out.view(out.size(0),-1)
        return self.fc(out)



class VAE(nn.Module):
    def __init__(self, z_dim):
        super(VAE,self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2*z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, mu, logvar):
        mu = self.bn(mu)
        std = torch.exp(logvar)
        eps = torch.randn_like(std)

        return mu + std

    def forward(self,x):
        # 32x8x8
        b,c,h = x.size()
        x = x.view(b,-1)

        z_q_mu, z_q_logvar = self.encode(x).chunk(2, dim=1)
        # reparameterize
        z_q = self.reparameterize(z_q_mu, z_q_logvar)
        out = self.decode(z_q)
        out = out.view(b,c,h)

        KL = torch.sum(0.5 * (z_q_mu.pow(2) + z_q_logvar.exp().pow(2) - 1) - z_q_logvar)

        return out, KL


class VAE_4x(nn.Module):
    def __init__(self, z_dim):
        super(VAE_4x,self).__init__()

        # 32x8x8
        self.encode = nn.Sequential(nn.Linear(512, 2*z_dim),
                                    )
        self.bn = nn.BatchNorm1d(z_dim)
        self.decode = nn.Sequential(nn.Linear(z_dim, 512),
                                    nn.BatchNorm1d(512),
                                    nn.ReLU(),
                                    nn.Linear(512, 512),
                                    )

    def reparameterize(self, k, l, m, n, a_mu, a_logvar, b_mu, b_logvar, c_mu, c_logvar, d_mu, d_logvar):
        a_mu = self.bn(a_mu)
        a_std = torch.exp(a_logvar)
        b_mu = self.bn(b_mu)
        b_std = torch.exp(b_logvar)
        c_mu = self.bn(c_mu)
        c_std = torch.exp(c_logvar)
        d_mu = self.bn(d_mu)
        d_std = torch.exp(d_logvar)

        mu = k * a_mu + l * b_mu + m * c_mu + n * d_mu
        a_var = torch.pow(a_std, 2)
        b_var = torch.pow(b_std, 2)
        c_var = torch.pow(c_std, 2)
        d_var = torch.pow(d_std, 2)

        std = torch.pow(k * a_var + l * b_var + m * c_var + n * d_var, 0.5)

        return mu + std

    def forward(self, k, l, m, n, a, b, c, d):
        # 32x8x8
        batch,cl,h = a.size()
        a = a.view(batch, -1)
        b = b.view(batch, -1)
        c = c.view(batch, -1)
        d = d.view(batch, -1)

        x = torch.cat((a,b,c,d), dim=0)
        mu, logvar = self.encode(x).chunk(2, dim=1)
        a_mu, b_mu, c_mu, d_mu = mu.chunk(4, dim=0)
        a_logvar, b_logvar, c_logvar, d_logvar = logvar.chunk(4, dim=0)
        # reparameterize
        z_q = self.reparameterize(k, l, m, n, a_mu, a_logvar, b_mu, b_logvar, c_mu, c_logvar, d_mu, d_logvar)
        out = self.decode(z_q)
        out = out.view(batch,cl,h)

        return out



class MulLayer(nn.Module):
    def __init__(self, z_dim, matrixSize=32):
        super(MulLayer,self).__init__()
        # self.snet = CNN_VAE(layer, z_dim, matrixSize)
        self.snet = CNN(matrixSize)
        self.cnet = CNN(matrixSize)
        self.VAE = VAE(z_dim=z_dim)
        self.matrixSize = matrixSize

        self.compress = nn.Conv2d(512,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,512,1,1,0)

        self.transmatrix = None

    def forward(self,cF,sF,trans=True):
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF.size()
        sFF = sF.view(sb,sc,-1)
        sMean = torch.mean(sFF,dim=2,keepdim=True)
        sMean, KL = self.VAE(sMean)
        sMean = sMean.unsqueeze(3)
        sMeanC = sMean.expand_as(cF)
        sMeanS = sMean.expand_as(sF)
        sF = sF - sMeanS


        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(sF)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix, KL
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out



class MulLayer_4x(nn.Module):
    def __init__(self, z_dim, matrixSize=32):
        super(MulLayer_4x,self).__init__()
        self.snet = CNN(matrixSize)
        self.VAE = VAE_4x(z_dim=z_dim)
        self.cnet = CNN(matrixSize)
        self.matrixSize = matrixSize

        self.compress = nn.Conv2d(512,matrixSize,1,1,0)
        self.unzip = nn.Conv2d(matrixSize,512,1,1,0)

        self.transmatrix = None

    def forward(self, k, l, m, n, cF, sF_1, sF_2, sF_3, sF_4, trans=True):
        cb,cc,ch,cw = cF.size()
        cFF = cF.view(cb,cc,-1)
        cMean = torch.mean(cFF,dim=2,keepdim=True)
        cMean = cMean.unsqueeze(3)
        cMean = cMean.expand_as(cF)
        cF = cF - cMean

        sb,sc,sh,sw = sF_1.size()
        sFF_1 = sF_1.view(sb,sc,-1)
        sMean_1 = torch.mean(sFF_1,dim=2,keepdim=True)

        sFF_2 = sF_2.view(sb,sc,-1)
        sMean_2 = torch.mean(sFF_2,dim=2,keepdim=True)

        sFF_3 = sF_3.view(sb,sc,-1)
        sMean_3 = torch.mean(sFF_3,dim=2,keepdim=True)

        sFF_4 = sF_4.view(sb,sc,-1)
        sMean_4 = torch.mean(sFF_4,dim=2,keepdim=True)

        sMean = self.VAE(k, l, m, n, sMean_1, sMean_2, sMean_3, sMean_4)
        sMean = sMean.unsqueeze(3)
        sF_1 = sF_1 - sMean.expand_as(sF_1)
        sF_2 = sF_2 - sMean.expand_as(sF_2)
        sF_3 = sF_3 - sMean.expand_as(sF_3)
        sF_4 = sF_4 - sMean.expand_as(sF_4)

        sMeanC = sMean.expand_as(cF)

        compress_content = self.compress(cF)
        b,c,h,w = compress_content.size()
        compress_content = compress_content.view(b,c,-1)

        if(trans):
            cMatrix = self.cnet(cF)
            sMatrix = self.snet(k * sF_1 + l*sF_2 + m*sF_3 + n*sF_4)

            sMatrix = sMatrix.view(sMatrix.size(0),self.matrixSize,self.matrixSize)
            cMatrix = cMatrix.view(cMatrix.size(0),self.matrixSize,self.matrixSize)
            transmatrix = torch.bmm(sMatrix,cMatrix)
            transfeature = torch.bmm(transmatrix,compress_content).view(b,c,h,w)
            out = self.unzip(transfeature.view(b,c,h,w))
            out = out + sMeanC
            return out, transmatrix
        else:
            out = self.unzip(compress_content.view(b,c,h,w))
            out = out + cMean
            return out





