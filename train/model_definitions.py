import torch.nn as nn


class ContrastivePretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int=32,
      device: str = "cuda:0"
  ):
    super(ContrastivePretrainedLearner, self).__init__()
    self.device = device

    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim),
    )

  def forward(self, x):
    return self.encoder(x)
  

class AEPretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int= 32,
      device: str = "cuda:0"
  ):
    super(AEPretrainedLearner, self).__init__()
    self.device = device

    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, latent_dim),
    )

    self.decoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
    )

  def forward(self, x):
    return self.decoder(self.encoder(x))
  
  def encode(self, x):
    return self.encoder(x)
  
