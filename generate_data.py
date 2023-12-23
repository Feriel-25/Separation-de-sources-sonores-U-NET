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

class GeneratorDataset(Dataset):
    def __init__(self, mus):
        self.mus = mus
        self.track_duration = (128 - 1) * 768 / 8192
        self.tracks=self.get_dataset()
    def __len__(self):
        return len(self.tracks)

    def __getitem__(self, idx):
        track =self.tracks[idx]
    
        # Process the mixture audio (X)
        X = self.process_audio(np.mean(track.audio.T, axis=0))
       
        # Process the isolated vocal track (Y)
        Y = self.process_audio(np.mean(track.targets['vocals'].audio.T, axis=0))
     
        return torch.tensor(X,dtype=torch.float32), torch.tensor(Y,dtype=torch.float32)

    def get_dataset(self):
        tracks=[]
        for track in self.mus.tracks:
            track_length = len(track.audio)  # Longueur du track en échantillons
            segment_length = int(self.track_duration * track.rate)  # Longueur de chaque segment en échantillons

            for start in range(0, track_length, segment_length):
                end_sample = start + segment_length
                if end_sample > track_length:
                    break

                # Créez une copie du track pour chaque segment
                segment_track = track
                segment_track.chunk_duration = self.track_duration
                segment_track.chunk_start = start / track.rate  # Convertir en secondes
                tracks.append(segment_track)
      
        return tracks

    def process_audio(self, audio):
        epsilon= 1e-10
        # Resample the audio to 8192 Hz
        audio_resampled = librosa.resample(audio, orig_sr=44100, target_sr=8192)

        # Compute the STFT
        stft = librosa.stft(audio_resampled, n_fft=1024, hop_length=768)
        spectrum, phase = librosa.magphase(stft)
        # Get magnitude from the complex-valued STFT
        magnitude = np.abs(spectrum)
        magnitude = magnitude[:-1, :]
        # Normalize the magnitude to the range [0, 1]
        magnitude_normalized = (magnitude - np.min(magnitude)) / (np.max(magnitude) - np.min(magnitude) + epsilon)

        return magnitude_normalized