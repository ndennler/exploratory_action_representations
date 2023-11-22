import pandas as pd
import torch
from typing import Callable

import numpy as np
from sklearn.model_selection import LeaveOneOut

import torch.nn as nn
from evaluation.dataloader import QueryDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch import optim

def get_pids_for_training(signal_modality: str, signal: str, train_only: bool = False):
  pids = []

  for pid in pd.read_csv('./data/evaluation/test_queries.csv')['pid'].unique():
    train_df = pd.read_csv('./data/evaluation/train_queries.csv')
    df = train_df.query(f'type == "{signal_modality}" and pid == {pid} and signal == "{signal}"')
    if len(df) > 0:
      test_df = pd.read_csv('./data/evaluation/test_queries.csv')
      df = test_df.query(f'type == "{signal_modality}" and pid == {pid} and signal == "{signal}"')
      if len(df) > 0 or train_only:
        pids.append(pid)
  
  return pids


def get_train_test_dataloaders(
    PID: int,
    signal_modality: str,
    signal: str,
    batch_size: int = 32,
    train_only: bool = False
):
  loo = LeaveOneOut()

  train_dataloaders = [] 
  test_dataloaders = []

  train_df = pd.read_csv('./data/evaluation/all_queries.csv')
  df = train_df.query(f'type == "{signal_modality}" and pid == {PID} and signal == "{signal}"')
  if len(df) < 1:
    raise Exception(f'not enough data for PID {PID}')
  
  dataset = QueryDataset(df, train=True, kind=signal_modality, transform=torch.Tensor)
  train_dataloader = DataLoader(dataset, batch_size=batch_size)

  if train_only:
    return [train_dataloader], [None]
  
  for train_index, test_index in loo.split(df):
    # Access the rows in the DataFrame using iloc
    train_set = df.iloc[train_index]
    dataset = QueryDataset(train_set, train=True, kind=signal_modality, transform=torch.Tensor)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    train_dataloaders.append(train_dataloader)

    test_set = df.iloc[test_index]
    dataset = QueryDataset(test_set, train=True, kind=signal_modality, transform=torch.Tensor)
    test_dataloader = DataLoader(dataset, batch_size=batch_size)
    test_dataloaders.append(test_dataloader)

  return train_dataloaders, test_dataloaders


def train_single_epoch(
    embedding_model: nn.Module,
    reward_model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer,
    epoch: int,
    device: str = 'cuda'
):
  # set model to training mode
  embedding_model.eval()
  embedding_model.to(device)
  reward_model.train()
  reward_model.to(device)
  train_loss = 0
  for batch_idx, (option1, option2, option3, choice) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    optimizer.zero_grad()

    r1 = reward_model(embedding_model.encode(option1))
    r2 = reward_model(embedding_model.encode(option2))
    r3 = reward_model(embedding_model.encode(option3))
    r4 = torch.zeros(r1.shape).to(device)
    rewards = torch.cat((r1,r2,r3,r4), 1)
    # compute loss
    loss = loss_fn(rewards, choice)

    loss.backward()
    train_loss += loss.item()
    optimizer.step()

    if batch_idx % 50 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(option1), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item() / len(option1)))

  print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))




def eval_model(
    embedding_model: nn.Module,
    reward_model: nn.Module,
    eval_fn: Callable,
    data_loader: DataLoader,
    device: str = 'cuda'):
  # set model to training mode
  embedding_model.eval()
  embedding_model.to(device)
  reward_model.eval()
  reward_model.to(device)

  eval_values = []
  for batch_idx, (option1, option2, option3, choice) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    r1 = reward_model(embedding_model.encode(option1))
    r2 = reward_model(embedding_model.encode(option2))
    r3 = reward_model(embedding_model.encode(option3))
    r4 = torch.zeros(r1.shape).to(device)
    rewards = torch.cat((r1,r2,r3,r4), 1).to(device)
    # compute loss
    eval_result = eval_fn(rewards, choice)

    eval_values.append(eval_result.item())

  return np.nanmean(eval_values)

def calc_reward(
    embedding_model: nn.Module,
    reward_model: nn.Module,
    data_loader: DataLoader,
    index: int,
    device: str = 'cuda'):
  
  embedding_model.eval()
  embedding_model.to(device)
  reward_model.eval()
  reward_model.to(device)

  option1= data_loader.dataset.get_single_item_by_index(index).to(device)
  reward = reward_model(embedding_model.encode(option1)).cpu().item()

  return reward