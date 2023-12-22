import torch 
from tqdm import tqdm
import torch.nn as nn 
from UNET import UNET
from UNET_encoder import Encoder
from UNET_decoder import Decoder
import torch
import musdb
import random
import librosa
import numpy as np

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Decoder()
decoder = Encoder()
model   = UNET(encoder,decoder,device).to(device)

# On charge les paramètres des modèles : 
model.load_state_dict(torch.load('saves/best_model.pth',map_location=torch.device(device))["model_state_dict"])

# On passe en mode évaluation :
model.eval()

# On recupere un exemple du test set :
mus_test = musdb.DB(root="/content/musdb18",subsets="test")
track = random.choice(mus_test.tracks)

# Define parameters
target_sr = 8192  # Target sampling rate
hop_length = 768  # Hop length for STFT
n_fft = 1024  # Number of FFT components
segment_length_sec = 11  # Length of the audio patch in seconds
segment_samples = target_sr * segment_length_sec  # Number of samples in the audio patch

# STFT to convert the time-domain audio signal to the time-frequency domain
stft_patches = []
for start_sample in range(0, len(track.audio.T), segment_samples):
    end_sample = start_sample + segment_samples
    if end_sample > len(track.audio.T):
        # If the last segment is shorter than the desired length, it can be padded or ignored
        # Here we choose to ignore it
        break

    # Take the mean across the stereo channels (if applicable) and apply STFT
    segment = np.mean(track.audio.T[start_sample:end_sample], axis=1)
    stft_segment = librosa.stft(segment, n_fft=n_fft, hop_length=hop_length)

    # Only keep 128 time frames (128 points)
    stft_segment = stft_segment[:, :128]
    
    # Discard the last frequency bin to make it 512 instead of 513 if necessary
    stft_segment = stft_segment[:-1, :] if stft_segment.shape[0] == 513 else stft_segment

    stft_patches.append(stft_segment)

D = librosa.stft(track, n_fft=n_fft, hop_length=hop_length)

# Compute the magnitude and phase
magnitude, phase = librosa.magphase(D)
#stft inverve to convert the time-frequency domain back to the time-domain audio signal
audio_patches = []
for stft_segment in stft_patches:
    # Invert the STFT
    audio_segment = librosa.istft(stft_segment, hop_length=hop_length)
    
    # Append the audio segment to the list of audio patches
    audio_patches.append(audio_segment)

# Concatenate the audio patches into a single audio signal
audio = np.concatenate(audio_patches)
audio = audio * phase
# Save the audio signal as a WAV file
librosa.output.write_wav('audio.wav', audio, sr=target_sr)