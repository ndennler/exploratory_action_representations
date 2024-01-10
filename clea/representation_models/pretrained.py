import torch.nn as nn
import torch 


class PretrainedEncoder(nn.Module):
  def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, latent_dim: int = 32, device: str = "cuda:0"):
    super(PretrainedEncoder, self).__init__()
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
  
  def encode(self, x):
    return self.encoder(x)
  

class PretrainedDecoder(nn.Module):
  def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, latent_dim: int = 32, device: str = "cuda:0"):
    super(PretrainedDecoder, self).__init__()
    self.device = device

    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
    )

  def forward(self, x):
    return self.decoder(x)




class ContrastiveAEPretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(ContrastiveAEPretrainedLearner, self).__init__()
    self.device = device

    self.encoder = PretrainedEncoder(input_dim, hidden_dim, latent_dim, device=device)

    self.decoder = PretrainedDecoder(input_dim, hidden_dim, latent_dim, device=device)

  def forward(self, x):
    return self.decoder(self.encoder(x))
  
  def encode(self, x):
    return self.encoder(x)

class ContrastivePretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int = 32,
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
  
  def encode(self, x):
    return self.encoder(x)
  

class AEPretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int = 32,
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
  

class VAEPretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(VAEPretrainedLearner, self).__init__()
    self.device = device
    self.nz = latent_dim

    self.encoder = nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, 2*latent_dim),
    )

    self.decoder = nn.Sequential(
        nn.Linear(latent_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.ReLU(),
        nn.Linear(hidden_dim, input_dim),
    )

  def forward(self, x):
    q = self.encoder(x)
    z = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
    return {'q': q, 'rec': self.decoder(z)}
  
  def encode(self, x):
    return self.encoder(x)[:,:self.nz]

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
    a_m, a_dev = a_q[:,:nz], a_q[:,nz:]
    p_m, p_dev = p_q[:,:nz], p_q[:,nz:]
    n_m, n_dev = n_q[:,:nz], n_q[:,nz:]
    desired_m, desired_dev = torch.zeros((a_q.shape[0], nz), device=self.device), torch.zeros((a_q.shape[0], nz), device=self.device)

    kl_loss = self.kl_divergence(a_m, a_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(p_m, p_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(n_m, n_dev, desired_m, desired_dev).mean()
    
    return rec_loss + beta * kl_loss

    
class RandomPretrainedLearner(nn.Module):

  def __init__(
      self,
      input_dim: int = 1024,
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RandomPretrainedLearner, self).__init__()
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
  
  def encode(self, x):
    return self.encoder(x)

class TaskEmbedder(nn.Module):
    def __init__(self, input_size, device):
        super(TaskEmbedder, self).__init__()
        self.input_size = input_size
        self.device = device
        
        self.task_embedding = nn.Embedding(num_embeddings=4, embedding_dim=input_size)
        self.linear = nn.Linear(2*self.input_size, self.input_size)
        
    def forward(self, embedding, task_idxs):
        task_embeds = self.task_embedding(task_idxs.to(self.device))
        combined = torch.cat([task_embeds, embedding], axis=1)
        return self.linear(combined)



