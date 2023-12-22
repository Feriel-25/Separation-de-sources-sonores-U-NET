import torch 
import torch.nn as nn
from UNET_decoder import Decoder
from UNET_encoder import Encoder

class UNET(nn.Module):
    def __init__(self,encoder,decoder,device):
        super (UNET,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device=device
        
    def forward(self,x):
        x1,x2,x3,x4,x5,x6 = self.encoder(x)
        mask = self.decoder(x1,x2,x3,x4,x5,x6)
        return mask