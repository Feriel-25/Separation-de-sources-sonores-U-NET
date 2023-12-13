import torch 
import tqdm
import torch.nn as nn 
from UNET import UNET
from UNET_encoder import Encoder
from UNET_decoder import Decoder


def train (model,device,dataloader,epochs,print_every,learning_rate=0.001): 
    loss_function = nn.L1Loss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 

    for epoch in range(epochs):
        for batch in tqdm(dataloader, desc=f'Epoch {epoch + 1}/{epochs}'): 
            X, Y = batch
            X, Y = X.to(device), Y.to(device)
            
            optimizer.zero_grad()  # Zero the gradients
            mask = unet(X)  # Forward pass
            predicted_spectrogram = mask * X  # Apply mask to input
            
            loss = loss_function(predicted_spectrogram, Y)  # Calculate loss
            loss.backward()  # Backward pass
            optimizer.step()  # Update weights

          
            if epoch % print_every == 0:
             print('Epoch: ', epoch + 1, 'loss =', '{:.6f}'.format(loss.item()))
            
    print('Training Finished')


device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataloader=...
encoder= Encoder()
decoder= Decoder()
unet = UNET(encoder, decoder, device).to(device)

epochs=...
print_every=...
learning_rate=...
batch= train(unet,device,dataloader,epochs,print_every,learning_rate)