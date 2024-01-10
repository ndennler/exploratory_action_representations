import pandas as pd
import torch
from typing import Callable

import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim

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
):
  # set model to training mode
  model.train()
  model.to(device)

  train_loss = 0
  for batch_idx, (anchor, positive, negative) in enumerate(data_loader):

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
    
    elif embedding_type in ['VAE']:
      loss = loss_fn(a_embed, p_embed, n_embed, anchor, positive, negative, beta=.01)

    elif embedding_type in ['contrastive+autoencoder']:
      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
      loss += nn.TripletMarginLoss()(model.encode(anchor),model.encode(positive),model.encode(negative))

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
  # set model to training mode
  model.train()
  model.to(device)

  task_embedder.train()
  task_embedder.to(device)

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
      a_embed = model.task_forward(anchor,task_idxs)
      p_embed = model.task_forward(positive, task_idxs)
      n_embed = model.task_forward(negative, task_idxs)
    

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

