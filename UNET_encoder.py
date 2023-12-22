unet_encoder.py 
import torch
import torch.nn as nn
import torch.nn.functional as F 

class Encoder(nn.Module):
    def _init_(self, kernel_size=5, stride=2):
        super(Encoder, self)._init_()
        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride,padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride,padding=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride,padding=2)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size, stride,padding=2)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, stride,padding=2)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size, stride,padding=2)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)),0.2)

        x2= F.leaky_relu(self.bn2(self.conv2(x1)),0.2)

        x3= F.leaky_relu(self.bn3(self.conv3(x2)),0.2)
        
        x4 = F.leaky_relu(self.bn4(self.conv4(x3)),0.2)
       
        x5 = F.leaky_relu(self.bn5(self.conv5(x4)),0.2)
       
        x6= F.leaky_relu(self.bn6(self.conv6(x5)),0.2)
        
        return x1,x2,x3,x4,x5,x6