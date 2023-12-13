import torch 
import torch.nn as nn
from UNET_decoder import Decoder
from UNET_encoder import Encoder

class UNET(nn.Module):
    def __init__(self,encoder,decoder):
        super (UNET,self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def forward(self,x):
        enc_out,concat_out = self.encoder(x)
        mask = self.decoder(enc_out,concat_out)
        return mask