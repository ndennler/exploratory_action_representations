import torch.nn as nn


class RewardLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      device: str = "cuda"
  ):
    super(RewardLearner, self).__init__()
    self.device = device

    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 1),
    )

  def forward(self, x):
    return self.encoder(x)