import torch
import torch.nn as nn

class CVAE(nn.module):
    def __init__(self):
        super().__init__()

        self.latent_size = latent_size
        self.encoder = Encoder(enc_layer_sizes, latent_size)
        self.decoder = Decoder(dec_layer_sizes, latent_size)

    def forward(self,x):

        mu, log_var = self.encoder(x)

    def sampling(self, mu, log_var):

    def inference(self, z):



class Encoder(nn.module):

class Decoder(nn.module):

