import pandas as pd
import torch
from typing import Callable

import torch.nn as nn
from train.dataloader import ChoiceDataset
from torch.utils.data import DataLoader
from torch import optim

from train.model_definitions import ContrastivePretrainedLearner, AEPretrainedLearner, RandomPretrainedLearner, ContrastiveAEPretrainedLearner


def get_dataloader(dataset_type: str, signal_type: str, batch_size: int = 32):

  df = pd.read_csv('./data/plays_and_options.csv')
  df = df.query(f'type == "{dataset_type}" & signal == "{signal_type}"')
  dataset = ChoiceDataset(df, train=True, kind=dataset_type, transform=torch.Tensor)
  embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

  return embedding_dataloader, dataset.get_input_dim()

def get_model(model_type, 
              input_dim: int = 1024,
              hidden_dim: int = 512,
              latent_dim: int = 64):
  
  if model_type == 'contrastive':
    return ContrastivePretrainedLearner(input_dim, hidden_dim, latent_dim)
  
  elif model_type == 'random':
    return RandomPretrainedLearner(input_dim, hidden_dim, latent_dim)

  elif model_type == 'autoencoder':
    return AEPretrainedLearner(input_dim, hidden_dim, latent_dim)
  
  elif model_type == 'contrastive+autoencoder':
    return ContrastiveAEPretrainedLearner(input_dim, hidden_dim, latent_dim)
  
  else:
    raise Exception(f'Model type {model_type} is not yet implemented!')


def train_single_epoch(
    embedding_type: str,
    model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer,
    epoch: int,
    device: str = 'cuda'
):
  # set model to training mode
  model.train()
  model.to(device)
  
  train_loss = 0
  for batch_idx, (anchor, positive, negative) in enumerate(data_loader):

    anchor = anchor.to(device)
    positive = positive.to(device)
    negative = negative.to(device)

    optimizer.zero_grad()

    a_embed = model(anchor)
    p_embed = model(positive)
    n_embed = model(negative)

    # compute loss
    if embedding_type in ['contrastive']:
      loss = loss_fn(a_embed, p_embed, n_embed)
    elif embedding_type in ['autoencoder']:
      loss = loss_fn(a_embed, anchor) + loss_fn(p_embed, positive) + loss_fn(n_embed, negative)
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


if __name__ == '__main__':
  # Define training variables, feel free to modify these for the problem
  log_interval = 100
  num_epochs = 100
  input_dim = 1024
  hidden_dim = 512
  latent_dim = 64
  batch_size = 32
  device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
    

