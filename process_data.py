import musdb
import random
import torch
mus = musdb.DB(root="C://Users//ferie//MUSDB18//MUSDB18-7")
               
for track in mus:
    print(track.name)
    print(track.stems[0]) #for mixture
    print(track.stems[4]) #for vocals

def generator():
    while True:
        track = random.choice(mus.tracks)
        track.chunk_duration = (128 - 1) * 768 / 8192
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x = track.audio.T
        y = track.targets['vocals'].audio.T
        yield x, y

def get_batch(generator, batch_size):
    x_batch = []
    y_batch = []
    for _ in range(batch_size):
        x, y = next(generator)
        x_batch.append(x)
        y_batch.append(y)
    return [torch.tensor(x_batch), torch.tensor(y_batch)]

def tfst(batches,window=1024, hop=768):
    for batch in batches:
        x = torch.stft(batch[0],window,hop)
        y = torch.stft(batch[1],window,hop)
        yield x,y

  
    
