
'''
sequence models
'''  

import torch
import torch.nn as nn
from clea.representation_models.pretrained import PretrainedEncoder

class RawSequenceEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=.2, device='cpu',):
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
    def __init__(self, input_size, hidden_size, num_layers, dropout=.2, device='cpu'):
        super(RawSequenceDecoder, self).__init__()
        self.device = device
        
        self.gru = nn.GRU(input_size, hidden_size//(num_layers*2), num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        self.linear = nn.Linear(hidden_size//2, input_size)
    
    def forward(self, x, hidden):
        output, hidden = self.gru(x, hidden.contiguous())
        output = self.linear(output)
        return output, hidden

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=.2, device='cpu',):
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
            output, hidden = self.decoder(input, hidden.contiguous())
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
          output, hidden = self.decoder(input, hidden.contiguous())
          output_seq.append(output)
          input = output
      
      # stack outputs along seq_len dimension and return
      output_seq = torch.cat(output_seq, dim=1)
      return output_seq
    


class Seq2SeqVAE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=.2, device='cpu'):
        super(Seq2SeqVAE, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.nz = hidden_size
        
        self.encoder = PretrainedEncoder(input_size, hidden_size*2, num_layers, dropout, device)
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
            output, hidden = self.decoder(input, hidden.contiguous())
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return {'q': q, 'rec': output_seq}
    
    def taskconditioned_forward(self, x, task_idxs):
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = x.size(1)
        
        #encoding phase
        self.encoder.init_hidden(batch_size)
        q = self.encoder(x)
      
        hidden = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
        hidden = self.task_embedder(hidden, task_idxs)

        hidden = hidden.reshape(2*self.num_layers, batch_size, self.hidden_size//(2*self.num_layers))
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden.contiguous())
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return {'q': q, 'rec': output_seq}
    
    def encode(self, x):
      return self.encoder(x)[:,:self.nz]
    
    def taskconditioned_encode(self, x, task_idxs, task_embedder):
      z = self.encoder(x)[:,:self.nz]
      return task_embedder(z, task_idxs)
    
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


    









class Pretrained2RawSeq2Seq(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_dim, num_layers, dropout=.2, device='cpu',):
        super(Pretrained2RawSeq2Seq, self).__init__()
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.seq_len = 50
        
        self.encoder = PretrainedEncoder(input_size, hidden_size, latent_dim, device)
        self.decoder = RawSequenceDecoder(output_size, latent_dim, num_layers, dropout, device)
        
    # def forward(self, x):
    #     # x has shape (batch_size, seq_len, input_size)
    #     batch_size = x.size(0)
    #     seq_len = self.seq_len
        
    #     #encoding phase
    #     hidden = self.encoder(x)
        
    #     # decoding phase
    #     input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
    #     output_seq = []

    #     for _ in range(seq_len):
    #         output, hidden = self.decoder(input, hidden)
    #         output_seq.append(output)
    #         input = output
        
    #     # stack outputs along seq_len dimension and return
    #     output_seq = torch.cat(output_seq, dim=1)
    #     return output_seq
    
    def encode(self, x):
      return self.encoder(x)
    
    def taskconditioned_forward(self, x, task_idxs, task_embedder):
        #encoding phase
        hidden = self.encoder(x)
        hidden = task_embedder(hidden, task_idxs)

        return hidden
        
    def taskconditioned_encode(self, x, task_idxs, task_embedder):
        z = self.encoder(x)
        return task_embedder(z, task_idxs)

    def decode(self, x):
        # x has shape (batch_size, seq_len, input_size)
        
        batch_size = x.size(0)
        seq_len = self.seq_len
        
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []
        hidden = x.reshape([batch_size, self.num_layers*2, -1]).transpose(1,0)

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden)
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq
    


class Pretrained2RawSeq2SeqVAE(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, latent_dim, num_layers, dropout=.2, device='cpu'):
        super(Pretrained2RawSeq2SeqVAE, self).__init__()
        self.input_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.nz = latent_dim
        
        self.seq_len = 50
        
        self.encoder = PretrainedEncoder(input_size, hidden_size, latent_dim*2, device)
        self.decoder = RawSequenceDecoder(output_size, latent_dim, num_layers, dropout, device)
        
    def taskconditioned_forward(self, x, task_idxs, task_embedder):
        q = self.encoder(x)
        z = q[:,:self.nz] + torch.exp(q[:, self.nz:]) * torch.randn([q.shape[0], self.nz], device=self.device)
        z = task_embedder(z, task_idxs)
        return {'q': q, 'rec': self.decode(z)}
    
    def taskconditioned_encode(self, x, task_idxs, task_embedder):
        z = self.encoder(x)[:,:self.nz]
        return task_embedder(z, task_idxs)
    
    def decode(self, x):
        # x has shape (batch_size, seq_len, input_size)
        batch_size = x.size(0)
        seq_len = self.seq_len
        
        # decoding phase
        input = torch.zeros((batch_size, 1, self.input_size)).to(x.device)
        output_seq = []
        hidden = x.reshape([batch_size, self.num_layers*2, -1]).transpose(1,0)

        for _ in range(seq_len):
            output, hidden = self.decoder(input, hidden)
            output_seq.append(output)
            input = output
        
        # stack outputs along seq_len dimension and return
        output_seq = torch.cat(output_seq, dim=1)
        return output_seq

    
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