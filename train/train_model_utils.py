import pandas as pd
import torch
from typing import Callable

import torch.nn as nn
from train.dataloader import ChoiceDataset, RawChoiceDataset, RawChoiceDatasetwithTaskEmbedding
from torch.utils.data import DataLoader
from torch import optim

from train.model_definitions import ContrastivePretrainedLearner, AEPretrainedLearner, VAEPretrainedLearner, RandomPretrainedLearner, ContrastiveAEPretrainedLearner

from train.model_definitions import RawImageEncoder, RawImageAE, RawImageVAE, RawSequenceEncoder, Seq2Seq, Seq2SeqVAE

'''

Methods for getting dataloaders and models

'''
def get_dataloader(dataset_type: str, signal_type: str, batch_size: int = 32, dataloader_type: str = 'pretrained_embedding'):
  '''
  returns a dataloader that contains all of the data for the given modality for pretraining representations.
  dataloader_type is one of { 'pretrained_embeddings' and 'raw_data' }
  dataset_type is one of {'visual', 'auditory', 'kinetic'}
  '''
  if dataloader_type == 'pretrained_embedding':
    df = pd.read_csv('./data/plays_and_options.csv')
    df = df.query(f'type == "{dataset_type}" & signal == "{signal_type}"')
    dataset = ChoiceDataset(df, train=True, kind=dataset_type, transform=torch.Tensor)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

    return embedding_dataloader, dataset.get_input_dim()

  elif dataloader_type == 'raw_data':
    df = pd.read_csv('./data/plays_and_options.csv')
    df = df.query(f'type == "{dataset_type}" & signal == "{signal_type}"')
    dataset = RawChoiceDataset(df, train=True, kind=dataset_type, transform=torch.Tensor)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

  elif dataloader_type == 'task_embedding':
    df = pd.read_csv('./data/plays_and_options.csv')
    df = df.query(f'type == "{dataset_type}"')
    dataset = RawChoiceDatasetwithTaskEmbedding(df, train=True, kind=dataset_type, transform=torch.Tensor)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=4)

    return embedding_dataloader, dataset.get_input_dim()

def get_model(model_type, 
              input_dim: int = 1024,
              hidden_dim: int = 512,
              latent_dim: int = 64,
              device: str = 'cpu'):
  
  if model_type == 'contrastive':
    return ContrastivePretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)
  
  elif model_type == 'random':
    return RandomPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)

  elif model_type == 'autoencoder':
    return AEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)

  elif model_type == 'VAE':
    return VAEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)
  
  elif model_type == 'contrastive+autoencoder':
    return ContrastiveAEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)
  
  else:
    raise Exception(f'Model type {model_type} is not yet implemented!')




def get_raw_data_model(model_type,
              kind,
              input_dim: list,
              hidden_dim: int = 512,
              latent_dim: int = 64,
              device: str = 'cpu'):
  
  if model_type in ['contrastive', 'random']:
    if kind in ['visual', 'auditory']:
      return RawImageEncoder(input_dim, hidden_dim, latent_dim, device=device)
    else:
      return RawSequenceEncoder(input_size=3, hidden_size=latent_dim, num_layers=2, dropout=0, device=device)
    
  if model_type in ['autoencoder', 'contrastive+autoencoder']:
    if kind in ['visual', 'auditory']:
      return RawImageAE(input_dim, kind, hidden_dim, latent_dim, device=device)
    else:
      return Seq2Seq(input_size=3, hidden_size=latent_dim, num_layers=2, dropout=.2, device=device)
    
  if model_type in ['VAE']:
    if kind in ['visual', 'auditory']:
      return RawImageVAE(input_dim, kind, hidden_dim, latent_dim, device=device)
    else:
      return Seq2SeqVAE(input_size=3, hidden_size=latent_dim, num_layers=2, dropout=.2, device=device)
  
  else:
    raise Exception(f'Model type {model_type} is not yet implemented!')




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

    # print(anchor)

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

    if batch_idx % 50 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(anchor), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

  print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
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

    
    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)

    # print(anchor)

    a_embed = model.encode(anchor)
    p_embed = model.encode(positive)
    n_embed = model.encode(negative)
    

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

    if batch_idx % 50 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(anchor), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

  print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
  return epoch*len(data_loader.dataset), train_loss / len(data_loader.dataset)

