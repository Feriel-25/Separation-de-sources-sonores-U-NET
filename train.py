
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
import librosa
import numpy as np


from torch.utils.data import Dataset


class NaiveGeneratorDataset(Dataset):
    def __init__(self, mus):
        self.mus = mus
        self.track_duration = (128 - 1) * 768 / 8192

    def __len__(self):
        return len(self.mus.tracks)

    def __getitem__(self, idx):
        track = self.mus.tracks[idx]
        track.chunk_duration = self.track_duration
        track.chunk_start = random.uniform(0, track.duration -self.track_duration)
        
        # Process the mixture audio (X)
        X = self.process_audio(np.mean(track.audio.T, axis=0))
        
        # Process the isolated vocal track (Y)
        Y = self.process_audio(np.mean(track.targets['vocals'].audio.T, axis=0))
        
        return torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)

    def process_audio(self, audio):
        epsilon= 1e-10
        
        # Resample the audio to 8192 Hz
        audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=8192)
        
        # Compute the STFT
        stft = librosa.stft(audio_resampled, n_fft=1024, hop_length=768)

        # Get magnitude from the complex-valued STFT
        magnitude = np.abs(stft)
        magnitude = magnitude[:-1, :]
        # Normalize the magnitude to the range [0, 1]
        magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + epsilon)   
        
        return magnitude_normalized




def train (model,device,dataloader,epochs,print_every,learning_rate=0.001): 
    loss_function = nn.L1Loss( reduction='sum')  
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
    best_loss=float('inf')
    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            X, Y = batch
            X, Y = X.to(device), Y.to(device)
        
            optimizer.zero_grad()  # Zero the gradients
            
            mask = model(X.unsqueeze(1)).float()  # Forward pass
          
            predicted_spectrogram = mask * X  # Apply mask to input
            
            loss = loss_function(predicted_spectrogram, Y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

          
            if epoch % print_every == 0:
             print('Epoch: ', epoch + 1, 'loss =', '{:.6f}'.format(loss.item()))
            
            if loss<best_loss : 
              torch.save({'model_state_dict': model,
              'optimizer_state_dict': optimizer.state_dict()},
              "/content/save/best_model.pth")

              best_loss=loss
    print('Training Finished')


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder= Encoder()
decoder= Decoder()
unet = UNET(encoder, decoder, device).to(device)

epochs=50
print_every=1
learning_rate=0.001

#mus = musdb.DB(root="/content/musdb18")
mus_train = musdb.DB(root="/content/musdb18",subsets="train")
#mus_train = musdb.DB(root="/content/drive/My Drive/musdb_dataset",subsets="train")
#mus_test = musdb.DB(subsets="test")
print("Loading data...")
generator_dataset = NaiveGeneratorDataset(mus_train)  

dataloader = DataLoader(generator_dataset, batch_size=32, shuffle=True)
print("Data loaded.")

print("Training...")
train(unet,device, dataloader,epochs,print_every,learning_rate)