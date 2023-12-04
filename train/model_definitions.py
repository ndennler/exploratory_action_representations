import torch.nn as nn
import torch 

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




'''
image models
'''

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
      type: str,
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

    padding = [(1,0), (0,2)] if type == 'auditory' else [(1,1), (0,0)]

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
      type: str, 
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RawImageAE, self).__init__()
    self.device = device

    self.encoder = RawImageEncoder(input_dim, hidden_dim, latent_dim, device)
    self.decoder = RawImageDecoder(type, self.encoder.get_intermediate_size(), self.encoder.get_reshape_size(), hidden_dim, latent_dim, device)

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
      type: str, 
      hidden_dim: int = 256,
      latent_dim: int = 32,
      device: str = "cuda:0"
  ):
    super(RawImageVAE, self).__init__()
    self.device = device
    self.nz = latent_dim

    self.encoder = RawImageEncoder(input_dim, hidden_dim, 2*latent_dim, device)
    self.decoder = RawImageDecoder(type, self.encoder.get_intermediate_size(), self.encoder.get_reshape_size(), hidden_dim, latent_dim, device)

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
    a_m, a_dev = a_q[:,:self.nz], a_q[:,self.nz:]
    p_m, p_dev = p_q[:,:self.nz], p_q[:,self.nz:]
    n_m, n_dev = n_q[:,:self.nz], n_q[:,self.nz:]
    desired_m, desired_dev = torch.zeros((a_q.shape[0], nz), device=self.device), torch.zeros((a_q.shape[0], nz), device=self.device)

    kl_loss = self.kl_divergence(a_m, a_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(p_m, p_dev, desired_m, desired_dev).mean() + \
              self.kl_divergence(n_m, n_dev, desired_m, desired_dev).mean()
    
    return rec_loss + beta * kl_loss
  



'''
sequence models
'''  
class RawSequenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RawSequenceEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size 
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.gru = nn.GRU(input_size, hidden_size//(num_layers*2), num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        
    def forward(self, x):
        
        self.init_hidden(x.shape[0])
        output, final_hidden = self.gru(x,self.h0)
        # print(final_hidden.shape)
        # print(final_hidden.transpose(0,1).reshape(-1, self.hidden_size).shape)
        # return output[:, -1, :]
        return final_hidden.transpose(0,1).reshape(-1, self.hidden_size)
    
    def forward_for_seq2seq(self, x):
        self.init_hidden(x.shape[0])
        output, final_hidden = self.gru(x,self.h0)
        # print(output[:,-1,:].shape)
        return output, final_hidden
    
    def init_hidden(self, batch_size):
        self.h0 = torch.randn(self.num_layers*2, batch_size, self.hidden_size//(self.num_layers*2), device=self.device)
    
    def encode(self, x):
       return self.forward(x)



class RawSequenceDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(RawSequenceDecoder, self).__init__()
        self.device = device
        
        self.gru = nn.GRU(input_size, hidden_size//(num_layers*2), num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size//2, input_size)
    
    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden)
        output = self.linear(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(Seq2Seq, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        
        self.encoder = RawSequenceEncoder(input_size, hidden_size, num_layers, dropout, device)
        self.decoder = RawSequenceDecoder(input_size, hidden_size, num_layers, dropout, device)
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #encoding phase
        self.encoder.init_hidden(batch_size)
        _, hidden = self.encoder.forward_for_seq2seq(x)
        
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden)
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq
    
    def encode(self, x):
      self.batch_size = x.size(0)
      self.seq_len = x.size(1)
      return self.encoder(x)

    def decode(self, x):
      input = torch.zeros((self.batch_size, 1, self.input_size)).to(x.device) 
      hidden = x.reshape([self.batch_size, self.num_layers*2, -1]).transpose(1,0)
      output_seq = []

      for _ in range(self.seq_len):
          output, hidden = self.decoder(input, hidden)
          output_seq.append(output)
          input = output
      
      # stack outputs along seq_len dimension and return
      output_seq = torch.cat(output_seq, dim=1)
      return output_seq
    


class Seq2SeqVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, device):
        super(Seq2SeqVAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.nz = hidden_size
        
        self.encoder = RawSequenceEncoder(input_size, hidden_size*2, num_layers, dropout, device)
        self.decoder = RawSequenceDecoder(input_size, hidden_size, num_layers, dropout, device)
        
    def forward(self, x):
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #encoding phase
        self.encoder.init_hidden(batch_size)
        q = self.encoder(x)
        hidden = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
        
        hidden = hidden.reshape(2*self.num_layers, batch_size, self.hidden_size//(2*self.num_layers))
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden)
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return {'q': q, 'rec': output_seq}
    
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
      a_m, a_dev = a_q[:,:self.nz], a_q[:,self.nz:]
      p_m, p_dev = p_q[:,:self.nz], p_q[:,self.nz:]
      n_m, n_dev = n_q[:,:self.nz], n_q[:,self.nz:]
      desired_m, desired_dev = torch.zeros((a_q.shape[0], nz), device=self.device), torch.zeros((a_q.shape[0], nz), device=self.device)

      kl_loss = self.kl_divergence(a_m, a_dev, desired_m, desired_dev).mean() + \
                self.kl_divergence(p_m, p_dev, desired_m, desired_dev).mean() + \
                self.kl_divergence(n_m, n_dev, desired_m, desired_dev).mean()
      
      return rec_loss + beta * kl_loss



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
    