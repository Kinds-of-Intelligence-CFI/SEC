import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Convolutional Autoencoder as visual processing of the agent (generating compressed representations).
class Conv_Autoencoder(nn.Module):
    def __init__(self, n_hidden=1000):
        '''
        Convolutional autoencoder in PyTorch, prepared to process images of shape (84,84,3). A sparsity constraint can be added to the middle layer.

        Args:
            n_hidden (int; default=100): number of hidden units in the middle layer.
        '''
        super().__init__()

        self.n_hidden = n_hidden
        self.dim1, self.dim2 = 10, 10

        # Encoder
        self.conv1 = nn.Conv2d(3, 16, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(64 * self.dim1 * self.dim2, n_hidden)

        # Decoder
        self.fc2 = nn.Linear(n_hidden, 64 * self.dim1 * self.dim2)
        self.conv4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1, output_padding=0)
        self.conv6 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1, output_padding=0)

    def encoder(self, x):
        # Encoder
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        #x = x.view(-1, 64 * self.dim1 * self.dim2)
        x = x.reshape(-1, 64 * self.dim1 * self.dim2)
        x = F.relu(self.fc1(x))
        return x

    def decoder(self, x):
        # Decoder
        x = F.relu(self.fc2(x))
        #x = x.view(-1, 64, self.dim1, self.dim2)
        x = x.reshape(-1, 64, self.dim1, self.dim2)
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.sigmoid(self.conv6(x))
        return x

    def forward(self, x):
        h = self.encoder(x)
        out = self.decoder(h)
        return out, h

    def backward(self, optimizer, criterion, x, y_true, alpha=0):

        optimizer.zero_grad()

        y_pred, hidden = self.forward(x)

        recon_loss = criterion(y_pred, y_true)

        orthonorm_loss = 0
        if alpha != 0:
            batch_size, hidden_dim = hidden.shape
            I = torch.eye(hidden_dim, device='cuda')
            H = torch.mm(hidden.t(), hidden)
            orthonorm_loss = alpha * torch.norm(I - H)/(batch_size*hidden_dim)   # change to /(hidden_dim**2) ??

        loss = recon_loss + orthonorm_loss
        loss.backward()

        optimizer.step()

        return recon_loss.item()

    def get_embedding(self, obs):
        #if obs.shape[2] <= 3:
        #obs = np.transpose(obs, (2,0,1))
        #print('obs shape: ', obs.shape)
        if obs.shape[-1] <= 3:
            obs = np.transpose(obs, (0,3,1,2))
        #if obs.shape[2] <= 3:
        #   obs = np.transpose(obs, (2,0,1))
        #print('obs shape 2: ', obs.shape)
        #obs_tensor = torch.from_numpy(obs)
        #obs_tensor = obs_tensor.permute(0, 3, 1, 2)
        #print('input_tensor shape: ', obs_tensor.shape)
        x = torch.Tensor(obs).to('cuda')
        #embedding = self.encoder(x).cpu().numpy()
        embedding = self.encoder(x).detach().cpu().numpy()

        return embedding


class PerceptualLayer(object):

    def __init__(self, gameID, aeID, prototype_length=20):

        self.prototype_length = prototype_length
        self.reconstruct_error = 100
        print("prototype_length :", self.prototype_length)
        #self.reconstructed_img = np.zeros((84,84,3))

        #file_path = os.path.abspath('./data/autoencoders/'+gameID+'/'+aeID+'_p'+str(self.prototype_length)+'.pth')
        file_path = os.path.abspath('./data/autoencoders/'+gameID+'_'+aeID+'_'+str(self.prototype_length)+'hidden.pth')

        if os.path.exists(file_path):
            self.model = Conv_Autoencoder(n_hidden=self.prototype_length).to('cuda')
            self.model.load_state_dict(torch.load(file_path))
            print('FILE '+gameID+' '+aeID+'_p'+str(self.prototype_length)+'.pth LOADED')
        else:
            print('FILE DOES NOT EXIST')


    def load_model(self, path, prototype_length):  # n_hidden = 100 or 1000, according to the filename
        model = Conv_Autoencoder(n_hidden=prototype_length).to('cuda')
        model.load_state_dict(torch.load(path))

        return model


    def get_prototype(self, img):
        prototype = self.model.get_embedding(img)
        #flattened_prototype = prototype.flatten()

        return prototype

