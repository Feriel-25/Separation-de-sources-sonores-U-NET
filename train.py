from tqdm import tqdm
import torch.nn as nn
from UNET import UNET
from UNET_encoder import Encoder
from UNET_decoder import Decoder
from torch.utils.data import DataLoader
import torch
import musdb
from generate_data import GeneratorDataset


def train (model,device,dataloader,epochs,print_every,learning_rate=0.001):
    
    loss_function = nn.L1Loss( reduction='sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    best_loss=float('inf')

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'):
            X, Y = batch
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            mask = model(X.unsqueeze(1)).float()  # Forward pass
            mask=mask.squeeze(1)
            predicted_spectrogram = mask * X  # Apply mask to input

            loss = loss_function(predicted_spectrogram, Y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights
            
            if epoch % print_every == 0:
             print('Epoch: ', epoch + 1, 'loss =', '{:.6f}'.format(loss.item()))

            if loss<best_loss :
              torch.save(model.state_dict(), "/content/save/best_model.pth")
              best_loss=loss
    print('Training Finished')


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder= Encoder()
decoder= Decoder()
unet = UNET(encoder, decoder, device).to(device)

epochs=30
print_every=1
learning_rate=0.1

print("Loading data...")
mus_train = musdb.DB(root="/content/musdb18",subsets="train")
generator_dataset = GeneratorDataset(mus_train)
dataloader = DataLoader(generator_dataset, batch_size=32, shuffle=True)
print("Data loaded.")

print("Training...")
train(unet,device, dataloader,epochs,print_every,learning_rate)