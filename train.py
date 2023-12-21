import torch 
from tqdm import tqdm
import torch.nn as nn 
from UNET import UNET
from UNET_encoder import Encoder
from UNET_decoder import Decoder
from torch.utils.data import DataLoader
import torch
from torch.utils.data import IterableDataset
import musdb
import random


from torch.utils.data import Dataset
"""class NaiveGeneratorDataset(IterableDataset):
    def __init__(self, batch_size, track_duration=5.0):
        self.mus = musdb.DB(root="C://Users//linda//OneDrive//Documents//M2 SORBONNE//SON av//TP4//musdb18")
        self.batch_size = batch_size
        self.track_duration = track_duration

    def __iter__(self):
        return self

    def __next__(self):
        batch_x, batch_y = [], []
        for _ in range(self.batch_size):
            track = random.choice(self.mus.tracks)
            track.chunk_duration = self.track_duration
            track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
            x = torch.tensor(track.audio.T)
            y = torch.tensor(track.targets['vocals'].audio.T)
            batch_x.append(x)
            batch_y.append(y)
        return torch.stack(batch_x), torch.stack(batch_y)"""

class NaiveGeneratorDataset(Dataset):
    def __init__(self, total_samples, track_duration=5.0):
        self.mus = musdb.DB(root="C://Users//linda//OneDrive//Documents//M2 SORBONNE//SON av//TP4//musdb18")
        self.total_samples = total_samples
        self.track_duration = track_duration

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        track = random.choice(self.mus.tracks)
        track.chunk_duration = self.track_duration
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x = torch.tensor(track.audio.T)  
        y = torch.tensor(track.targets['vocals'].audio.T)
        return torch.tensor(x), torch.tensor(y)



def train (model,device,dataloader,epochs,print_every,learning_rate=0.001): 
    loss_function = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            X, Y = batch
            X, Y = X.to(device), Y.to(device)
            breakpoint()
            optimizer.zero_grad()  # Zero the gradients
            mask = unet(X)  # Forward pass
            breakpoint()
            predicted_spectrogram = mask * X  # Apply mask to input
            
            loss = loss_function(predicted_spectrogram, Y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

          
            if epoch % print_every == 0:
             print('Epoch: ', epoch + 1, 'loss =', '{:.6f}'.format(loss.item()))
            
    print('Training Finished')


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder= Encoder()
decoder= Decoder()
unet = UNET(encoder, decoder, device).double().to(device)

epochs=1
print_every=1
learning_rate=0.001

print("Loading data...")
generator_dataset = NaiveGeneratorDataset(total_samples=1000)  

dataloader = DataLoader(generator_dataset, batch_size=64, shuffle=True)

print("Data loaded.")
print("Training...")
train(unet,device, dataloader,epochs,print_every,learning_rate)