import pandas as pd
import torch
from typing import Callable

import numpy as np

import torch.nn as nn
from dataloader import QueryDataset
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torch import optim


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

    optimizer.zero_grad()

    r1 = reward_model(embedding_model(option1))
    r2 = reward_model(embedding_model(option2))
    r3 = reward_model(embedding_model(option3))
    r4 = torch.zeros(r1.shape)
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




def eval_single_epoch(
    embedding_model: nn.Module,
    reward_model: nn.Module,
    eval_fn: Callable,
    data_loader: DataLoader,
    device: str = 'cuda'
):
  # set model to training mode
  embedding_model.eval()
  reward_model.eval()

  eval_values = []
  for batch_idx, (option1, option2, option3, choice) in enumerate(data_loader):

    option1= option1.to(device)
    option2 = option2.to(device)
    option3 = option3.to(device)

    optimizer.zero_grad()

    r1 = reward_model(embedding_model(option1))
    r2 = reward_model(embedding_model(option2))
    r3 = reward_model(embedding_model(option3))
    r4 = torch.zeros(r1.shape)
    rewards = torch.cat((r1,r2,r3,r4), 1)
    # compute loss
    eval_result = eval_fn(rewards, choice)

    eval_values.append(eval_result.item())

  return np.nanmean(eval_values)






# Define training variables, feel free to modify these for the problem
log_interval = 100
num_epochs = 10
input_dim = 1024
hidden_dim = 512
latent_dim = 64
batch_size = 32
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = nn.Identity()
embedding_model = torch.load('../models/visual_contrastive.pth')

embedding_model = FeatureLearner(
  input_dim=input_dim,
  hidden_dim=hidden_dim,
  latent_dim=latent_dim,
  device=device
)
print(device)

kind = 'visual'
train_df = pd.read_csv('../data/train_queries.csv')
test_df = pd.read_csv('../data/test_queries.csv')

results = []

for pid in test_df['pid'].unique():
  
  print(pid)

  df = train_df.query(f'type == "{kind}" and pid == {pid}')

  # print(df, train_df['pid'])
  if len(df) < 2:
    continue

  dataset = QueryDataset(df, train=True, kind='visual', transform=torch.Tensor)
  embedding_dataloader = DataLoader(dataset, batch_size=batch_size)


  # build model
  rl = RewardLearner(
    input_dim=latent_dim,
    hidden_dim=hidden_dim,
    device=device
  )

  # put model on device
  rl.to(device)

  # device optimizer
  optimizer = optim.Adam(rl.parameters())
  loss_fn = nn.CrossEntropyLoss()

  # train
  for epoch in range(1, num_epochs + 1):
    train_single_epoch(
      # embedding_model=nn.Identity(),
      embedding_model=embedding_model,
      reward_model=rl,
      loss_fn=loss_fn,
      data_loader=embedding_dataloader,
      optimizer=optimizer,
      epoch=epoch,
      device=device
    )


  df = test_df.query(f'type == "{kind}" and pid == {pid}')
  dataset = QueryDataset(df, train=True, kind='visual', transform=torch.Tensor)
  embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

  # evaluate
  accuracy = eval_single_epoch(
                embedding_model=embedding_model,
                reward_model=rl,
                eval_fn=Accuracy(task="multiclass", num_classes=4),
                data_loader=embedding_dataloader,
                device=device
            )

  results.append(accuracy)

print(f'Accuracy is {np.nanmean(results)}')

  

