from UNET import UNET
from UNET_encoder import Encoder
from UNET_decoder import Decoder
import torch
import musdb
import librosa
import numpy as np
import soundfile as sf

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder = Encoder()
decoder = Decoder()


model = UNET(encoder, decoder, device).to(device)
model.load_state_dict(torch.load('/content/save/best_model.pth'))
model.eval()


n_fft = 1024
hop_length = 768
track_duration = (128 - 1) * 768 / 8192

mus_test = musdb.DB(root="/content/musdb18", subsets="test")
track = mus_test.tracks[4]

track_length = len(track.audio) # Longueur du track en échantillons
segment_length = int(track_duration * track.rate)  # Longueur de chaque segment en échantillons


reconstructed_audio=[]
for start in range(0, track_length, segment_length):
    end_sample = start + segment_length
    if end_sample > track_length:
        break
    
    track.chunk_duration = track_duration
    track.chunk_start = start / track.rate

    # Mixage des pistes
    audio_mix = np.mean(track.audio.T, axis=0)
    audio_mix_resampled = librosa.resample(audio_mix, orig_sr=44100, target_sr=8192)

    # STFT du mixage
    stft_mix = librosa.stft(audio_mix_resampled, n_fft=n_fft, hop_length=hop_length)

    # On récupère la magnitude et la phase du mixage
    magnitude_mix, phase_mix = librosa.magphase(stft_mix)
    magnitude_mix = magnitude_mix[:-1, :]
    phase_mix = phase_mix[:-1, :]

    # On normalise la magnitude pour le modèle
    magnitude_mix_normalized = (magnitude_mix - np.min(magnitude_mix)) / (np.max(magnitude_mix) - np.min(magnitude_mix) + 1e-10)

    # Prédiction de la magnitude des voix séparées
    mask = model(torch.tensor(magnitude_mix_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device))
    mask= mask.squeeze(0).squeeze(0).detach().cpu().numpy()

    # Reconstruction de la magnitude des voix séparées
    D_voices = mask * np.exp(1j * phase_mix)

    # Reconstruction de l'audio des voix séparées
    audio_voices = librosa.istft(D_voices, hop_length=hop_length)
    reconstructed_audio.append(audio_voices)

# On concatène les segments pour obtenir l'audio séparé complet
reconstructed_audio=np.concatenate(reconstructed_audio, axis=0)

# On écrit l'audio séparé dans un fichier
sf.write('voices_separated.wav', reconstructed_audio, 44100)

