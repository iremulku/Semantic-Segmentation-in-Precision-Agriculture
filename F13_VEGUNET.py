# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:07:22 2020

@author: Erdem
"""

from __future__ import print_function
import torch 
import torch.nn as nn
from torch.autograd import gradcheck
from functools import partial
import torch.nn.functional as F
nonlinearity = partial(F.relu,inplace=True)
  
class Reciprocal(nn.Module):
    
    def __init__(self):
        super(Reciprocal, self).__init__()
    
    def forward(self, input):
        # See the autograd section for explanation of what happens here.
        self.input = input
        return torch.div(1,input)
 
    def backward(self,grad_value):
        return -grad_value*torch.div(1,torch.mul(self.input,self.input))

class VegIndBlock(nn.Module):
    def __init__(self,channel):
        super(VegIndBlock, self).__init__()
        self.onebyone = nn.Conv2d(channel, 3, kernel_size=1)
        self.onebytwo = nn.Conv2d(3, 1, kernel_size=1)
        self.recip = Reciprocal()

    def forward(self, input):
        onybyone_out = self.onebyone(input)
        reciproc_out = self.recip(onybyone_out)
        concat_out = torch.cat(onybyone_out,reciproc_out,dim=1)
        onebytwo_out = self.onebytwo(concat_out)
        return onebytwo_out

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        fsize= 5
        psize = int((fsize-1)/2)
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, fsize,padding=(psize, psize)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, fsize,padding=(psize, psize)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2), nn.Dropout(0.5),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()
        self.bilinear = bilinear

        self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)
        
        self.drop = nn.Dropout(0.5)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        if self.bilinear:
            x1 = F.interpolate(x1, scale_factor=2, mode='bilinear', align_corners=True)
        else:
            x1 = self.up(x1)

        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])


        x = torch.cat([x2, x1], dim=1)
        x = self.drop(x)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class sigmoidOut(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(sigmoidOut, self).__init__()
        self.outsig= nn.Sigmoid()

    def forward(self, x):
        x = self.outsig(x)
        return x


class VegUNet(nn.Module):
    def __init__(self, classes, channels):
        super(VegUNet, self).__init__()
        
        self.veg = VegIndBlock(channels)
        self.inc = inconv(1, 64)
        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)
        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)
        self.outc = outconv(64, classes)
        self.outsig = sigmoidOut(1,1)

    def forward(self, x):
        x0 = self.veg(x)
        x1 = self.inc(x0)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.outsig(x)
        #return F.sigmoid(x)
        return x


##############################################################################   
if __name__ == '__main__':
    dev = "cuda:0"  
    device = torch.device(dev)
    model = VegUNet(classes=1,channels=5).to(device)
        
    input = (torch.randn(1,5,224,224,dtype=torch.double,requires_grad=True))
    #print(input)
    layer = VegIndBlock(channel=5);
    test = gradcheck(layer, input, eps=1e-6, atol=1e-4)
    print(test)