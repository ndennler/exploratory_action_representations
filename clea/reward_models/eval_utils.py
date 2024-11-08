import pandas as pd
import torch
from typing import Callable
from tqdm import tqdm
import numpy as np
from sklearn.model_selection import LeaveOneOut

from PIL import Image

import os
import torch.nn as nn
from clea.dataloaders.query_loaders import QueryDataset, RawQueryDataset, CachedRawQueryDataset, CachedRawQueryTaskEmbeddingDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch import optim

def get_pids_for_training(signal_modality: str, signal: str, train_only: bool = False):
  pids = []

  for pid in pd.read_csv('../data/evaluation/test_queries.csv')['pid'].unique():
    train_df = pd.read_csv('../data/evaluation/train_queries.csv')

    df = train_df.query(f'type == "{signal_modality}" and pid == {pid} and signal == "{signal}"')

    if signal == "all_signals":
      df = train_df.query(f'type == "{signal_modality}" and pid == {pid}')

    if len(df) > 0:
      test_df = pd.read_csv('../data/evaluation/test_queries.csv')
      df = test_df.query(f'type == "{signal_modality}" and pid == {pid} and signal == "{signal}"')
      
      if signal == "all_signals":
        df = test_df.query(f'type == "{signal_modality}" and pid == {pid}')

      if len(df) > 0 or train_only:
        pids.append(pid)
  
  return pids


def get_dataloaders(
    PID: int,
    modality: str,
    conditioning: str,
    embedding_path:str,
):
  loo  = LeaveOneOut()
  train_dataloaders = []
  test_dataloaders = []

  query_df = pd.read_csv('../data/evaluation/all_queries.csv')

  query = f'type == "{modality}" and pid == {PID}'

  if conditioning == 'independent':
    for sig in ['idle', 'searching', 'has_information', 'has_item']:

      df = query_df.query(f'{query} and signal == "{sig}"')

      if len(df) > 1: # add all queries of the signal if there are enough to do loo with
        for train_index, test_index in loo.split(df):
          train_set = df.iloc[train_index]
          test_set = df.iloc[test_index]

          dataset = CachedRawQueryTaskEmbeddingDataset(train_set, train=True, transform=torch.Tensor, name=embedding_path)
          train_dataloader = DataLoader(dataset, batch_size=32)
          train_dataloaders.append(train_dataloader)

          dataset = CachedRawQueryTaskEmbeddingDataset(test_set, train=True, transform=torch.Tensor, name=embedding_path)
          test_dataloader = DataLoader(dataset, batch_size=32)
          test_dataloaders.append(test_dataloader)
        

  elif conditioning == 'taskconditioned':
    df = query_df.query(query)
    if len(df) <= 1:
      return [], []

    for train_index, test_index in loo.split(df):
      train_set = df.iloc[train_index]
      test_set = df.iloc[test_index]

      dataset = CachedRawQueryTaskEmbeddingDataset(train_set, train=True, transform=torch.Tensor, name=embedding_path)
      train_dataloader = DataLoader(dataset, batch_size=32)
      train_dataloaders.append(train_dataloader)

      dataset = CachedRawQueryTaskEmbeddingDataset(test_set, train=True, transform=torch.Tensor, name=embedding_path)
      test_dataloader = DataLoader(dataset, batch_size=32)
      test_dataloaders.append(test_dataloader)

  return train_dataloaders, test_dataloaders
    

  





def get_train_test_dataloaders(
    PID: int,
    signal_modality: str,
    signal: str,
    batch_size: int = 32,
    train_only: bool = False,
    raw_data: bool = False,
    model_name: str = None
):
  loo = LeaveOneOut()

  train_dataloaders = [] 
  test_dataloaders = []

  train_df = pd.read_csv('../data/evaluation/all_queries.csv')
  df = train_df.query(f'type == "{signal_modality}" and pid == {PID} and signal == "{signal}"')

  if signal == "all_signals":
      df = train_df.query(f'type == "{signal_modality}" and pid == {PID}')

  if len(df) < 1:
    raise Exception(f'not enough data for PID {PID}')
  
  if raw_data:
    dataset = CachedRawQueryDataset(df, train=True, transform=torch.Tensor, name=model_name)
    if signal == 'all_signals':
      dataset = CachedRawQueryTaskEmbeddingDataset(df, train=True, transform=torch.Tensor, name=model_name)
  
  else:
    dataset = QueryDataset(df, train=True, kind=signal_modality, transform=torch.Tensor)
  train_dataloader = DataLoader(dataset, batch_size=batch_size)

  if train_only:
    return [train_dataloader], [None]
  
  for train_index, test_index in loo.split(df):
    # Access the rows in the DataFrame using iloc
    train_set = df.iloc[train_index]

    if raw_data:
      dataset = RawQueryDataset(train_set, train=True, transform=torch.Tensor, name=model_name)
    else:
      dataset = QueryDataset(train_set, train=True, kind=signal_modality, transform=torch.Tensor)
    train_dataloader = DataLoader(dataset, batch_size=batch_size)
    train_dataloaders.append(train_dataloader)

    test_set = df.iloc[test_index]
    if raw_data:
      dataset = RawQueryDataset(test_set, train=True, transform=torch.Tensor, name=model_name)
    else:
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
  if hasattr(embedding_model, 'encoder'): #clean this up with a model.move_to_device
        embedding_model.encoder.device = device
  reward_model.train()
  reward_model.to(device)
  train_loss = 0
  for batch_idx, (option1, option2, option3, choice, option_idxs) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    optimizer.zero_grad()

    # comment out if we are using raw data
    option1 = embedding_model.encode(option1)
    option2 = embedding_model.encode(option2)
    option3 = embedding_model.encode(option3)

    r1 = reward_model(option1)
    r2 = reward_model(option2)
    r3 = reward_model(option3)

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
  # print(precomputed_embeds)
  # return precomputed_embeds
  

def train_reward_model_one_epoch(
    reward_model: nn.Module,
    loss_fn: Callable,
    data_loader: DataLoader,
    optimizer,
    device: str = 'cuda'):
  
  reward_model.train()
  reward_model.to(device)
  train_loss = 0

  for batch_idx, (option1, option2, option3, choice, option_idxs) in enumerate(data_loader):
      
      option1= option1.to(device)
      option2 = option2.to(device)
      option3 = option3.to(device)
      choice = choice.to(device)
  
      optimizer.zero_grad()
  
      r1 = reward_model(option1)
      r2 = reward_model(option2)
      r3 = reward_model(option3)
  
      r4 = torch.zeros(r1.shape).to(device)
      rewards = torch.cat((r1,r2,r3,r4), 1)
      # compute loss
      loss = loss_fn(rewards, choice)
  
      loss.backward()
      train_loss += loss.item()
      optimizer.step()

  return train_loss

def evaluate_reward_model(
    reward_model: nn.Module,
    eval_fn: Callable,
    data_loader: DataLoader,
    device: str = 'cuda'):
  
  # set model to evaluate mode
  reward_model.eval()
  reward_model.to(device)

  eval_values = []
  tasks = []
  for batch_idx, (option1, option2, option3, choice, _) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    r1 = reward_model(option1)
    r2 = reward_model(option2)
    r3 = reward_model(option3)

    r4 = torch.zeros(r1.shape).to(device)
    rewards = torch.cat((r1,r2,r3,r4), 1).to(device)
    # compute loss
    eval_result = eval_fn(rewards, choice)

    eval_values.append(eval_result.item())

    
  return np.nanmean(eval_values)
    
  





def train_single_epoch_task_embeds(
    embedding_model: nn.Module,
    task_embedder: nn.Module,
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
  for batch_idx, (option1, option2, option3, choice, option_idxs) in enumerate(data_loader):

    # print(option1.shape)
    # input('shapes ok?')

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    optimizer.zero_grad()

    # comment out if we are using raw data
    # option1 = embedding_model.encode(option1)
    # option2 = embedding_model.encode(option2)
    # option3 = embedding_model.encode(option3)

    # option1 = task_embedder(option1, task_idxs)
    # option2 = task_embedder(p_embed, task_idxs)
    # option3 = task_embedder(n_embed, task_idxs)

    r1 = reward_model(option1)
    r2 = reward_model(option2)
    r3 = reward_model(option3)

    r4 = torch.zeros(r1.shape).to(device)
    rewards = torch.cat((r1,r2,r3,r4), 1)
    # compute loss
    loss = loss_fn(rewards, choice)

    loss.backward()
    train_loss += loss.item()
    optimizer.step()

    # if batch_idx % 50 == 0:
    #   print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
    #         epoch, batch_idx * len(option1), len(data_loader.dataset),
    #         100. * batch_idx / len(data_loader), loss.item() / len(option1)))

  # print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))
  # print(precomputed_embeds)
  # return precomputed_embeds
  



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
  tasks = []
  for batch_idx, (option1, option2, option3, choice, _) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)
    choice = choice.to(device)

    r1 = reward_model(embedding_model.encode(option1))
    r2 = reward_model(embedding_model.encode(option2))
    r3 = reward_model(embedding_model.encode(option3))

    # r1 = reward_model(option1)
    # r2 = reward_model(option2)
    # r3 = reward_model(option3)

    r4 = torch.zeros(r1.shape).to(device)
    rewards = torch.cat((r1,r2,r3,r4), 1).to(device)
    # compute loss
    eval_result = eval_fn(rewards, choice)

    eval_values.append(eval_result.item())
    # print(task)
    # tasks.append(task.item())

    
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


def generate_embeddings(model, model_str, kind, embedding_size, device: str = 'cuda'):
  
  if os.path.exists(f'../data/embeds/{model_str}.npy'):
    return
  
  train_df = pd.read_csv('../data/evaluation/all_queries.csv').query(f'type=="{kind}"')
  dataset = RawQueryDataset(train_df, train=True, kind=kind, transform=torch.Tensor)
  train_dataloader = DataLoader(dataset, batch_size=32)

  model.eval()
  model.to(device)

  array = np.zeros((len(dataset.stimulus_mapping), embedding_size))
  
  for batch_idx, (option1, option2, option3, choice, option_idxs) in enumerate(train_dataloader):
    print(batch_idx)
    
    embeds = model.encode(option1.to(device))
    for i,em in zip(option_idxs[0], embeds):
      array[int(i), :] = em.detach().cpu().numpy()

    embeds = model.encode(option2.to(device))
    for i,em in zip(option_idxs[1], embeds):
      array[int(i), :] = em.detach().cpu().numpy()

    embeds = model.encode(option3.to(device))
    for i,em in zip(option_idxs[2], embeds):
      array[int(i), :] = em.detach().cpu().numpy()


def generate_embeddings_task_conditioned(model, task_embedder, model_str, kind, embedding_size, device: str = 'cuda'):
  if os.path.exists(f'../data/embeds/{model_str}.npy'):
    print(f'embeddings already exists: {model_str}')
    return
  
  train_df = pd.read_csv('../data/evaluation/all_queries.csv').query(f'type=="{kind}"')
  dataset = RawQueryDataset(train_df, train=True, kind=kind, transform=torch.Tensor, task_embedding=True)
  train_dataloader = DataLoader(dataset, batch_size=32)

  model.eval()
  model.to(device)

  if 'VAE' in model_str:
    model.task_embedder.eval()
    model.task_embedder.to(device)
    print(model.task_embedder.device, model.device)
    
  task_embedder.eval()
  task_embedder.to(device)

  array = np.zeros((len(dataset.stimulus_mapping), 4, embedding_size))
  
  for batch_idx, (option1, option2, option3, choice, option_idxs, task_index) in enumerate(train_dataloader):
    tasks = task_index.detach().cpu().numpy()
    print(batch_idx)
    if 'VAE' in model_str:
      embeds = model.task_encode(option1.to(device), task_index.to(device))
    else:
      embeds = model.encode(option1.to(device))
      embeds = task_embedder(embeds, task_index)

    for i,t,em in zip(option_idxs[0], task_index, embeds):
      array[int(i), t, :] = em.detach().cpu().numpy()

    if 'VAE' in model_str:
      embeds = model.task_encode(option2.to(device), task_index.to(device))
    else:
      embeds = model.encode(option2.to(device))
      embeds = task_embedder(embeds, task_index)
    for i,t,em in zip(option_idxs[1], task_index, embeds):
      array[int(i), t, :] = em.detach().cpu().numpy()

    if 'VAE' in model_str:
      embeds = model.task_encode(option3.to(device), task_index.to(device))
    else:
      embeds = model.encode(option3.to(device))
      embeds = task_embedder(embeds, task_index)
    for i,t,em in zip(option_idxs[2], task_index, embeds):
      array[int(i), t, :] = em.detach().cpu().numpy()
    

  np.save(f'../data/embeds/{model_str}.npy', array)

  # raise Exception('break!')





def generate_all_embeddings_taskconditioned(model, task_embedder, dataframe, device, data_dir='../data', pretrained_embeds_array=None):
  model.eval()
  model.to(device)
  task_embedder.eval()
  task_embedder.to(device)

  out_size = task_embedder.input_size
  embeds = np.zeros((dataframe['id'].max() + 1, 4, out_size))

  print(embeds.shape)

  for signal_index in range(4):
    for i, row in tqdm(dataframe.iterrows()):
      id = row['id']

      if pretrained_embeds_array is not None:
        x = torch.Tensor(pretrained_embeds_array[id]).unsqueeze(0).to(device)
        out = model.encode(x)
        out = task_embedder(out, torch.Tensor([signal_index]).int().to(device))
        embeds[id, signal_index, :] = out.detach().cpu().numpy()
        

      elif row['type'] == 'Video':
        im= np.array(Image.open(f"{data_dir}/visual/vis/{row['file'].replace('mp4', 'jpg')}")) / 255.0
        im = np.moveaxis(im, -1, 0)
        out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))
        out = task_embedder(out, torch.Tensor([signal_index]).int().to(device))
        embeds[id, signal_index, :] = out.detach().cpu().numpy()

      elif row['type'] == 'Audio':
        im= np.array(Image.open(f"{data_dir}/auditory/aud/{row['file'].replace('wav', 'jpg')}")) / 255.0
        im = np.moveaxis(im, -1, 0)
        out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))
        out = task_embedder(out, torch.Tensor([signal_index]).int().to(device))
        embeds[id, signal_index, :] = out.detach().cpu().numpy()

      elif row['type'] == 'Movement':
        trajectories = np.load(f"{data_dir}/kinetic/behaviors.npy")
        traj = trajectories[id] * 25
        out = model.encode(torch.Tensor(traj).unsqueeze(0).to(device))
        out = task_embedder(out, torch.Tensor([signal_index]).int().to(device))
        embeds[id, signal_index, :] = out.detach().cpu().numpy()
        
      else:
        raise Exception(f'Unknown Stimulus Type: {row["type"]}')
      
  return embeds
      

    
def generate_all_embeddings_independent(model, dataframe, embedding_size, signal, device, data_dir='../data', pretrained_embeds_array=None, embed_storage_path=None):
  model.eval()
  model.to(device)
  TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

  out_size = embedding_size
  if embed_storage_path is not None:
    embeds = np.load(embed_storage_path)
  else:
    embeds = np.zeros((dataframe['id'].max() + 1, 4, out_size))

  print(embeds.shape)

  for i, row in tqdm(dataframe.iterrows()):
    id = row['id']
    signal_index = TASK_INDEX_MAPPING[signal]

    if pretrained_embeds_array is not None:
      x = torch.Tensor(pretrained_embeds_array[id]).unsqueeze(0).to(device)
      out = model.encode(x)
      embeds[id, signal_index, :] = out.detach().cpu().numpy()
      

    elif row['type'] == 'Video':
      im= np.array(Image.open(f"{data_dir}/visual/vis/{row['file'].replace('mp4', 'jpg')}")) / 255.0
      im = np.moveaxis(im, -1, 0)
      out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))
      embeds[id, signal_index, :] = out.detach().cpu().numpy()

    elif row['type'] == 'Audio':
      im= np.array(Image.open(f"{data_dir}/auditory/aud/{row['file'].replace('wav', 'jpg')}")) / 255.0
      im = np.moveaxis(im, -1, 0)
      out = model.encode(torch.Tensor(im).unsqueeze(0).to(device))
      embeds[id, signal_index, :] = out.detach().cpu().numpy()

    elif row['type'] == 'Movement':
      trajectories = np.load(f"{data_dir}/kinetic/behaviors.npy")
      traj = trajectories[id] * 25
      out = model.encode(torch.Tensor(traj).unsqueeze(0).to(device))
      embeds[id, signal_index, :] = out.detach().cpu().numpy()
      
    else:
      raise Exception(f'Unknown Stimulus Type: {row["type"]}')
    
  return embeds