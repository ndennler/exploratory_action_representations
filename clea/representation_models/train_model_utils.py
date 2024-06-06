import pandas as pd
import torch
from typing import Callable

import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim


class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)

class _RepeatSampler(object):
    """ Sampler that repeats forever.

    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

'''
Methods for Training
'''

def train_single_epoch(
    embedding_type: str,
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer,
    epoch: int,
    device: str = 'cuda',
    margin: float = 1
):

  train_loss = 0
  for batch_idx, (anchor, positive, negative) in enumerate(data_loader):
    # print(anchor.shape)
    optimizer.zero_grad()

    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)

    # print(anchor.shape)

    a_embed = model(anchor)
    p_embed = model(positive)
    n_embed = model(negative)

    # compute loss
    if embedding_type in ['contrastive']:
      loss = loss_fn(a_embed, p_embed, n_embed)
    
    elif embedding_type in ['autoencoder']:
      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
      loss +=  0.01*torch.norm(a_embed, p=1) + 0.01*torch.norm(p_embed, p=1) + 0.01*torch.norm(n_embed, p=1)
    
    elif embedding_type in ['VAE']:
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)

    elif embedding_type in ['contrastive+autoencoder']:
      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
      loss += nn.TripletMarginLoss(margin=margin)(model.encode(anchor),model.encode(positive),model.encode(negative))

    elif embedding_type in ['contrastive+VAE']:
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)
      loss += nn.TripletMarginLoss(margin=margin)(model.encode(anchor),model.encode(positive),model.encode(negative))

    loss.backward()
    train_loss += loss.item()
    optimizer.step()

    # if batch_idx % 50 == 0:
      # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
      #       epoch, batch_idx * len(anchor), len(data_loader.dataset),
      #       100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

    # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
    return epoch*len(data_loader.dataset), train_loss / len(data_loader.dataset)




def train_single_epoch_with_task_embedding(
    embedding_type: str,
    model: nn.Module,
    task_embedder: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    model_optimizer,
    embed_optimizer,
    epoch: int,
    device: str = 'cuda',
):

  train_loss = 0
  for batch_idx, (anchor, positive, negative, task_idxs) in enumerate(data_loader):

    model_optimizer.zero_grad()
    embed_optimizer.zero_grad()

    task_idxs = task_idxs.to(device)
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)

    # print(anchor)
    if embedding_type in ['contrastive', 'autoencoder', 'contrastive+autoencoder']:
      a_embed = model.encode(anchor)
      p_embed = model.encode(positive)
      n_embed = model.encode(negative)

    elif embedding_type in ['VAE']:
      a_embed = model.taskconditioned_forward(anchor, task_idxs, task_embedder)
      p_embed = model.taskconditioned_forward(positive, task_idxs, task_embedder)
      n_embed = model.taskconditioned_forward(negative, task_idxs, task_embedder)
    

    # compute loss
    if embedding_type in ['contrastive']:
      a_embed = task_embedder(a_embed, task_idxs)
      p_embed = task_embedder(p_embed, task_idxs)
      n_embed = task_embedder(n_embed, task_idxs)

      loss = loss_fn(a_embed, p_embed, n_embed)
    
    elif embedding_type in ['autoencoder']:
      a_embed = task_embedder(a_embed, task_idxs)
      p_embed = task_embedder(p_embed, task_idxs)
      n_embed = task_embedder(n_embed, task_idxs)

      a_embed = model.decode(a_embed)
      p_embed = model.decode(p_embed)
      n_embed = model.decode(n_embed)

      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
    
    elif embedding_type in ['VAE']:
      #TODO: figure out how to get this to work
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)

    elif embedding_type in ['contrastive+autoencoder']:
      a_embed = task_embedder(a_embed, task_idxs)
      p_embed = task_embedder(p_embed, task_idxs)
      n_embed = task_embedder(n_embed, task_idxs)

      loss = nn.TripletMarginLoss()(a_embed, p_embed, n_embed)

      a_embed = model.decode(a_embed)
      p_embed = model.decode(p_embed)
      n_embed = model.decode(n_embed)

      loss += loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
      

    loss.backward()
    train_loss += loss.item()
    model_optimizer.step()
    embed_optimizer.step()

    # if batch_idx % 50 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #         epoch, batch_idx * len(anchor), len(data_loader.dataset),
    #         100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

  # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
  return epoch*len(data_loader.dataset), train_loss / len(data_loader.dataset)


def train_single_epoch_with_task_embedding_from_pretrained(
    embedding_type: str,
    model: nn.Module,
    task_embedder: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    model_optimizer,
    embed_optimizer,
    epoch: int,
    device: str = 'cuda',
):

  train_loss = 0
  for batch_idx, (a, p, n, anchor, positive, negative, task_idxs) in enumerate(data_loader):

    model_optimizer.zero_grad()
    embed_optimizer.zero_grad()

    task_idxs = task_idxs.to(device)
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)
    a,p,n = a.to(device), p.to(device), n.to(device)

    # print(anchor)
    if embedding_type in ['contrastive', 'autoencoder', 'contrastive+autoencoder']:
      a_embed = task_embedder(model.encode(a), task_idxs)
      p_embed = task_embedder(model.encode(p), task_idxs)
      n_embed = task_embedder(model.encode(n), task_idxs)

    elif embedding_type in ['VAE', 'contrastive+VAE']:
      a_embed = model.taskconditioned_forward(a, task_idxs, task_embedder)
      p_embed = model.taskconditioned_forward(p, task_idxs, task_embedder)
      n_embed = model.taskconditioned_forward(n, task_idxs, task_embedder)
    

    # compute loss
    if embedding_type in ['contrastive']:
      loss = loss_fn(a_embed, p_embed, n_embed)
    
    elif embedding_type in ['autoencoder']:
      a_embed = model.decode(a_embed)
      p_embed = model.decode(p_embed)
      n_embed = model.decode(n_embed)

      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
    
    elif embedding_type in ['VAE']:
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)

    elif embedding_type in ['contrastive+autoencoder']:
      loss = nn.TripletMarginLoss()(a_embed, p_embed, n_embed)

      a_embed = model.decode(a_embed)
      p_embed = model.decode(p_embed)
      n_embed = model.decode(n_embed)

      loss += loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
    
    elif embedding_type in ['contrastive+VAE']:
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)
      loss += nn.TripletMarginLoss()(a_embed['q'][:model.nz], p_embed['q'][:model.nz], n_embed['q'][:model.nz])
      

    loss.backward()
    train_loss += loss.item()
    model_optimizer.step()
    embed_optimizer.step()

    # if batch_idx % 50 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #         epoch, batch_idx * len(anchor), len(data_loader.dataset),
    #         100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

  # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
  return epoch*len(data_loader.dataset), train_loss / len(data_loader.dataset)

