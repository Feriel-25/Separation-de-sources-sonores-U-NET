import torch 
import torch.nn as nn

class Decoder (nn.Module):
    def __init__(self,stride=2,kernel_size=5,padding=2):
        super (Decoder,self).__init__()
        self.deconv1 = nn.ConvTranspose2d(512,256,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn1=nn.BatchNorm2d(256)
        self.deconv2 = nn.ConvTranspose2d(256,128,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn2=nn.BatchNorm2d(128)
        self.deconv3 = nn.ConvTranspose2d(128,64,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn3=nn.BatchNorm2d(64)
        self.deconv4 = nn.ConvTranspose2d(64,32,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn4=nn.BatchNorm2d(32)
        self.deconv5 = nn.ConvTranspose2d(32,16,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn5=nn.BatchNorm2d(16)
        self.deconv6 = nn.ConvTranspose2d(16,1,kernel_size=kernel_size,stride=stride,padding=padding)
        self.bn6=nn.BatchNorm2d(1)
        
    def forward(self,enc_out,concat_out):

        
        x = self.deconv1(enc_out)
        x=  nn.ReLU(self.bn1(x))
        x=  nn.Dropout2d(x)

        x = torch.cat((x,concat_out[0]),dim=1)
        x = self.deconv2(x)
        x=  nn.ReLU(self.bn2(x))
        x=  nn.Dropout2d(x)

        x = torch.cat((x,concat_out[1]),dim=1)
        x = self.deconv3(x)  
        x=  nn.ReLU(self.bn3(x))
        x=  nn.Dropout2d(x)

        x = torch.cat((x,concat_out[2]),dim=1)
        x = self.deconv4(x)
        x=  nn.ReLU(self.bn4(x))

        x = torch.cat((x,concat_out[3]),dim=1)
        x = self.deconv5(x)
        x=  nn.ReLU(self.bn5(x))

        x = torch.cat((x,concat_out[4]),dim=1)
        x = self.deconv6(x)
        mask = x.sigmoid(x)
        

        return mask
    