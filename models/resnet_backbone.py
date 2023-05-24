import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import math, copy 
from einops import rearrange

class single_conv(nn.Module):
    def __init__(self,ch_in,ch_out):
        super(single_conv,self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(ch_in, ch_out, kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
         
    def forward(self,x):
        x = self.conv(x)
        return x

class up_conv(nn.Module):
    def __init__(self,ch_in,ch_out, scale_factor=2):
        super(up_conv,self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor),
            nn.Conv3d(ch_in,ch_out,kernel_size=3,stride=1,padding=1),
            nn.BatchNorm3d(ch_out),
            nn.ReLU(inplace=True)
        )
        
    def forward(self,x):
        x = self.up(x)
        return x

def conv3x3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        if downsample is not None:
            self.downsample = nn.Sequential(
                nn.Conv3d(inplanes, planes,  kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

        self.stride = stride
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=2, feature_size=48):
        super(ResNetEncoder, self).__init__()
        self.in_channels = in_channels
        
        self.Maxpool = nn.MaxPool3d(kernel_size=2,stride=2)
        self.Conv = nn.Conv3d(in_channels, feature_size//2,  kernel_size=7, stride=1, padding=5)

        self.Conv1 = nn.Sequential(
            single_conv(feature_size//2, feature_size),
            BasicBlock(feature_size, feature_size, stride=1, downsample=None)
        )
        self.Conv2 = nn.Sequential(
            single_conv(feature_size, feature_size*2),
            BasicBlock(feature_size*2, feature_size*2, stride=1, downsample=None)
        )
        self.Conv3 = nn.Sequential(
            single_conv(feature_size*2, feature_size*4),
            BasicBlock(feature_size*4, feature_size*4, stride=1, downsample=None)
        )
        self.Conv4 = nn.Sequential(
            single_conv(feature_size*4, feature_size*8),
            BasicBlock(feature_size*8, feature_size*8, stride=1, downsample=None)
        )


    def forward(self, x):
        # encoding path
        x0 = self.Conv(x)

        x1 = self.Maxpool(x0)
        x1 = self.Conv1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.Conv2(x2)
        
        x3 = self.Maxpool(x2)
        x3 = self.Conv3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.Conv4(x4)

        return x4


# ####
# if __name__=='__main__':
#     model = ResNetEncoder(in_channels=1, feature_size=48)
#     y = torch.zeros((1,1, 80, 96, 80))
#     out = model(y)
#     print(out.shape)  
       
