import torch
import torchvision.models as models
from torch import nn
from models.gcn_lib import Grapher as GCB 

import os
from models.network_fusion import interactive, interactive1_2, SpatialAttention

def mkdir(path):

    isExists = os.path.exists(path) # 判断路径是否存在，若存在则返回True，若不存在则返回False
    if not isExists: # 如果不存在则创建目录
        os.makedirs(path)
        return True
    else:
        return False

class Conv2dRe(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1,
                 groups=1, bias=True):
        super(Conv2dRe, self).__init__()
        self.bcr = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                      dilation=dilation, groups=groups, bias=bias),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.bcr(x)


class VGG16_deconv(torch.nn.Module):
    def __init__(self, num_classes=1):
        super(VGG16_deconv, self).__init__()
        vgg16bn_pretrained = models.vgg19_bn(pretrained=True)
        vgg16bn_pretrained_depth = models.vgg19_bn(pretrained=True)
        pool_list = [6, 13, 26, 39, 52]
        self.de_pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.de_pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.de_pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        for index in pool_list:
            vgg16bn_pretrained.features[index].return_indices = True
            vgg16bn_pretrained_depth.features[index].return_indices = True

        self.encoder1 = vgg16bn_pretrained.features[:6]
        self.pool1 = vgg16bn_pretrained.features[6]
        self.dencoder1 = vgg16bn_pretrained_depth.features[:6]
        self.dpool1 = vgg16bn_pretrained_depth.features[6]
        self.trans1 = interactive1_2(d_model=128, ratio=2)

        self.encoder2 = vgg16bn_pretrained.features[7:13]
        self.pool2 = vgg16bn_pretrained.features[13]
        self.dencoder2 = vgg16bn_pretrained_depth.features[7:13]
        self.dpool2 = vgg16bn_pretrained_depth.features[13]
        self.trans2 = interactive1_2(d_model=256, ratio=2)

        self.encoder3 = vgg16bn_pretrained.features[14:26]
        self.pool3 = vgg16bn_pretrained.features[26]
        self.dencoder3 = vgg16bn_pretrained_depth.features[14:26]
        self.dpool3 = vgg16bn_pretrained_depth.features[26]
        self.trans3 = interactive(n=6, d_model=256, heads=4, dropout=0.1, activation="relu", pos_feats=32, num_pos_feats=128, ratio=2)
        
        self.encoder4 = vgg16bn_pretrained.features[27:39]
        self.pool4 = vgg16bn_pretrained.features[39]
        self.dencoder4 = vgg16bn_pretrained_depth.features[27:39]
        self.dpool4 = vgg16bn_pretrained_depth.features[39]
        self.trans4 = interactive(n=8, d_model=512, heads=4, dropout=0.1, activation="relu", pos_feats=16, num_pos_feats=256, ratio=2)

        self.encoder5 = vgg16bn_pretrained.features[40:52]
        self.pool5 = vgg16bn_pretrained.features[52]
        self.dencoder5 = vgg16bn_pretrained_depth.features[40:52]
        self.dpool5 = vgg16bn_pretrained_depth.features[52]
        self.trans5 = interactive(n=8, d_model=512, heads=4, dropout=0.1, activation="relu", pos_feats=8,
                                 num_pos_feats=256, ratio=2)
    
        self.cr5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, padding=1)
        )
        
        self.cr4 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1)
        )
        
        self.cr2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        
        self.cr1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1)
        )
        
        self.convl4 = nn.Sequential(
            Conv2dRe(1024, 512),
            Conv2dRe(512, 512)
        )
        
        self.convl2 = nn.Sequential(
            Conv2dRe(256, 256),
            Conv2dRe(256, 256)
        )
        
        self.convl3 = nn.Sequential(
            Conv2dRe(640, 640),
            Conv2dRe(640, 640)
        )
    
        self.gcb = nn.Sequential(GCB(640, 11, min(3 // 4 + 1, 18 // 11), 'mr', 'gelu', 'batch',
                                    'True', 'False', 0.2, 1, n=32*32//(4*4), drop_path=0.0,
                                    relative_pos=True, padding=5),
        )
        self.SA = SpatialAttention()
        
        self.reg_layer = nn.Sequential(
            nn.Conv2d(640, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, 1)
        )

    def forward(self, f):
        x = f[0]
        depthmap = f[1]
        encoder1 = self.encoder1(x)  # 64 256 256
        pool1, indices1 = self.pool1(encoder1)
        d1 = self.dencoder1(depthmap)
        d1, indicesd1 = self.dpool1(d1)
        out1 = self.trans1(pool1, d1)  # 64 128 128
       
        encoder2 = self.encoder2(pool1)  # 128 128 128
        pool2, indices2 = self.pool2(encoder2)
        d2 = self.dencoder2(d1)
        d2, indicesd2 = self.dpool2(d2)
        out2 = self.trans2(pool2, d2)  # 128 64 64
        
        encoder3 = self.encoder3(pool2)  # 256 64 64
        pool3, indices3 = self.pool3(encoder3)
        d3 = self.dencoder3(d2)
        d3, indicesd3 = self.dpool3(d3)
        out3 = self.trans3(pool3, d3)  # 256 32 32

        encoder4 = self.encoder4(pool3)
        pool4, indices4 = self.pool4(encoder4)
        d4 = self.dencoder4(d3)
        d4, indicesd4 = self.dpool4(d4)
        out4 = self.trans4(pool4, d4)

        encoder5 = self.encoder5(pool4)
        pool5, indices5 = self.pool5(encoder5)
        d5 = self.dencoder5(d4)
        d5, indicesd5 = self.dpool5(d5)
        out5 = self.trans5(pool5, d5)
        
#         print(out1.shape) # 64 128 128
#         print(out2.shape) # 128 64 64
#         print(out3.shape) # 256 32 32
#         print(out4.shape) # 512 16 16
#         print(out5.shape) # 512 8 8
        
        out_up = self.cr5(self.up1(out5)) # 512 16 16 -> 512 16 16
        out_up = self.convl4(torch.cat((out_up, out4), dim=1)) # 1024 16 16 -> 512 16 16
        
        out_up = self.cr4(self.up2(out_up)) # 512 32 32 -> 256 32 32
        
        out_down = self.cr1(self.de_pool1(out1)) # 64 128 128 -> 128 64 64
        out_down = self.convl2(torch.cat((out_down, out2), dim=1))  # 256 64 64 -> 256 64 64
        
        out_down = self.cr2(self.de_pool2(out_down)) # 256 32 32 -> 128 32 32

        out = torch.cat((out_down, out3), dim=1)  # 128+256 32 32
        out = self.convl3(torch.cat((out, out_up), dim=1)) # 128+512 32 32
        
        out = self.gcb(out)
        out = self.SA(out) * out + out
        out_features = self.reg_layer(out)
        
        # return torch.relu(out_features)
        return torch.relu(out_features)

class Imagemodel(nn.Module):
    def __init__(self):
        super(Imagemodel, self).__init__()
        self.model = VGG16_deconv()

    def forward(self, clip):
        return self.model(clip)
