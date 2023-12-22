
import torch
import torch.nn as nn

class Decoder(nn.Module):
    def _init_(self, stride=2, kernel_size=5, padding=2):
        super(Decoder, self)._init_()
        # Define the deconvolution and batch normalization layers
        self.deconv1 = nn.ConvTranspose2d(512, 256, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256*2, 128, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128*2, 64, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64*2, 32, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32*2, 16, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn5 = nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(16*2, 1, kernel_size=kernel_size, stride=stride, padding=padding,output_padding=1)
        self.bn6 = nn.BatchNorm2d(1)

        # Define ReLU, Dropout, and Sigmoid as layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout2d(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2, x3, x4, x5, x6):
        x = self.deconv1(x6)
        x = self.relu(self.bn1(x))
        x = self.dropout(x)
        
        x = torch.cat((x, x5), dim=1)
        x = self.deconv2(x)
        x = self.relu(self.bn2(x))
        x = self.dropout(x)

        x = torch.cat((x, x4), dim=1)
        x = self.deconv3(x)  
        x = self.relu(self.bn3(x))
        x = self.dropout(x)

        x = torch.cat((x, x3), dim=1)
        x = self.deconv4(x)
        x = self.relu(self.bn4(x))

        x = torch.cat((x, x2), dim=1)
        x = self.deconv5(x)
        x = self.relu(self.bn5(x))

        x = torch.cat((x, x1), dim=1)
        x = self.deconv6(x)
        mask = self.sigmoid(x)
        return mask