'''
image models
'''

import torch
import torch.nn as nn


class RawImageEncoder(nn.Module):

  def __init__(
      self,
      input_dim: list,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RawImageEncoder, self).__init__()
    self.device = device
    self.input_dim = input_dim

    self.conv_layers = nn.Sequential(
        nn.Conv2d(3, 16, kernel_size=(16,16), stride=(4, 4)),
        nn.LeakyReLU(),

        nn.Conv2d(16, 32, kernel_size=(8,8), stride=(4, 4)),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(),

        nn.Conv2d(32, 64, kernel_size=(4,4), stride=(2, 2)),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(),
    )
    self.to(self.device)
    intermediate_size = self.get_intermediate_size()

    self.flat_embeds = nn.Sequential(
        nn.Flatten(),

        nn.Linear(intermediate_size, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim),
    )
    

  def get_intermediate_size(self):
    return nn.Flatten()(self.conv_layers(torch.zeros([1, *self.input_dim], device=self.device))).size(1)
  
  def get_reshape_size(self):
    return self.conv_layers(torch.zeros([1, *self.input_dim], device=self.device)).shape[1:]

  def forward(self, x):
    return self.flat_embeds(self.conv_layers(x))
  
  def encode(self, x):
    return self.flat_embeds(self.conv_layers(x))


class RawImageDecoder(nn.Module):

  def __init__(
      self,
      intermediate_size: int,
      reshape_size: list,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RawImageDecoder, self).__init__()
    self.device = device
    self.reshape_size = reshape_size

    self.flat_embeds = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, intermediate_size),
    )

    padding = [(1,1), (0,0)]

    self.conv_transpose_layers = nn.Sequential(
        
        nn.BatchNorm2d(64),
        nn.ConvTranspose2d(64, 32, kernel_size=(4,4), stride=(2, 2)),
        nn.LeakyReLU(),

        nn.BatchNorm2d(32),
        nn.ConvTranspose2d(32, 16, kernel_size=(8,8), stride=(4, 4), output_padding=padding[0]),
        nn.LeakyReLU(),

        nn.ConvTranspose2d(16, 3, kernel_size=(16,16), stride=(4, 4), output_padding=padding[1]),
        nn.Sigmoid(),
    )    

  def forward(self, x):
    x = self.flat_embeds(x)
    return self.conv_transpose_layers(x.reshape(-1, *self.reshape_size))



class RawImageAE(nn.Module):

  def __init__(
      self,
      input_dim: list,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RawImageAE, self).__init__()
    self.device = device

    self.encoder = RawImageEncoder(input_dim, hidden_dim, latent_dim, device)
    self.decoder = RawImageDecoder(self.encoder.get_intermediate_size(), self.encoder.get_reshape_size(), hidden_dim, latent_dim, device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)
    return x
  
  def encode(self, x):
    return self.encoder(x)

  def decode(self, x):
    return self.decoder(x)
  


class RawImageVAE(nn.Module):

  def __init__(
      self,
      input_dim: list,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0",
      task_embedder: nn.Module = None
  ):
    super(RawImageVAE, self).__init__()
    self.device = device
    self.nz = latent_dim

    self.task_embedder = task_embedder
    self.encoder = RawImageEncoder(input_dim, hidden_dim, 2*latent_dim, device)
    self.decoder = RawImageDecoder(self.encoder.get_intermediate_size(), self.encoder.get_reshape_size(), hidden_dim, latent_dim, device)

  def forward(self, x):
    q = self.encoder(x)
    z = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
    return {'q': q, 'rec': self.decoder(z)}
  
  def task_forward(self, x, task_idxs):
    q = self.encoder(x)
    z = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
    z = self.task_embedder(z, task_idxs)
    return {'q': q, 'rec': self.decoder(z)}
  
  def encode(self, x):
    return self.encoder(x)[:,:self.nz]
  
  def task_encode(self, x, task_idxs):
    z = self.encoder(x)[:,:self.nz]
    return self.task_embedder(z, task_idxs)
  
  def kl_divergence(self, mu1, log_sigma1, mu2, log_sigma2):
    """Computes KL[p||q] between two Gaussians defined by [mu, log_sigma]."""
    return (log_sigma2 - log_sigma1) + (torch.exp(log_sigma1) ** 2 + (mu1 - mu2) ** 2) \
                / (2 * torch.exp(log_sigma2) ** 2) - 0.5
  
  def vae_loss(self, a_output, p_output, n_output, a_label, p_label, n_label, beta=1.0):

    a_q, a_recon = a_output['q'], a_output['rec']
    p_q, p_recon = p_output['q'], p_output['rec']
    n_q, n_recon = n_output['q'], n_output['rec']

    # compute reconstruction loss
    rec_loss = nn.MSELoss()(a_recon, a_label) + nn.MSELoss()(p_recon, p_label) + nn.MSELoss()(n_recon, n_label)

    # compute KL divergence loss
    nz = int(a_q.shape[-1]/2.0)
    a_m, a_dev = a_q[:,:self.nz], a_q[:,self.nz:]
    p_m, p_dev = p_q[:,:self.nz], p_q[:,self.nz:]
    n_m, n_dev = n_q[:,:self.nz], n_q[:,self.nz:]
    desired_m, desired_dev = torch.zeros((a_q.shape[0], nz), device=self.device), torch.zeros((a_q.shape[0], nz), device=self.device)

    kl_loss = self.kl_divergence(a_m, a_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(p_m, p_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(n_m, n_dev, desired_m, desired_dev).mean()
    
    return rec_loss + beta * kl_loss
  