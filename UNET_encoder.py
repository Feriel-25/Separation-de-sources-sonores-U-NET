import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, kernel_size, stride):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size, stride)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size, stride)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, kernel_size, stride)
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, kernel_size, stride)
        self.bn4 = nn.BatchNorm2d(128)
        self.conv5 = nn.Conv2d(128, 256, kernel_size, stride)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 512, kernel_size, stride)
        self.bn6 = nn.BatchNorm2d(512)
        
        self.leaky_relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        skip_connections =[]
        x = self.leaky_relu(self.bn1(self.conv1(x)))
        skip_connections .append(x)
        x = self.leaky_relu(self.bn2(self.conv2(x)))
        skip_connections .append(x)
        x = self.leaky_relu(self.bn3(self.conv3(x)))
        skip_connections .append(x)
        x = self.leaky_relu(self.bn4(self.conv4(x)))
        skip_connections .append(x)
        x = self.leaky_relu(self.bn5(self.conv5(x)))
        skip_connections .append(x)
        x = self.leaky_relu(self.bn6(self.conv6(x)))
        skip_connections .append(x)
        return torch.stack(skip_connections ),x
