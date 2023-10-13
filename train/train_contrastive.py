import pandas as pd
import torch
from typing import Callable

import torch.nn as nn
from dataloader import ChoiceDataset
from torch.utils.data import DataLoader
from torch import optim

from model_definitions import FeatureLearner


def train_single_epoch(
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
    loss = loss_fn(a_embed, p_embed, n_embed)

    loss.backward()
    train_loss += loss.item()
    optimizer.step()

    if batch_idx % 50 == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(anchor), len(data_loader.dataset),
            100. * batch_idx / len(data_loader), loss.item() / len(anchor)))

  print('====> Epoch: {} Average loss: {:.4f}'.format(epoch, train_loss / len(data_loader.dataset)))






# Define training variables, feel free to modify these for the problem
log_interval = 100
num_epochs = 100
input_dim = 1024
hidden_dim = 512
latent_dim = 64
batch_size = 32
device = "cpu" # "cuda" if torch.cuda.is_available() else "cpu"

print(device)

kind = 'visual'
df = pd.read_csv('../data/plays_and_options.csv')
df = df.query(f'type == "{kind}"')

dataset = ChoiceDataset(df,train=True, kind='visual', transform=torch.Tensor)

embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

# build model
fl = FeatureLearner(
  input_dim=input_dim,
  hidden_dim=hidden_dim,
  latent_dim=latent_dim,
  device=device
)

# put model on device
if torch.cuda.is_available():
  fl.cuda()

# device optimizer
optimizer = optim.Adam(fl.parameters())

loss_fn = nn.TripletMarginLoss()

# train
for epoch in range(1, num_epochs):
  train_single_epoch(
    model=fl,
    loss_fn=loss_fn,
    data_loader=embedding_dataloader,
    optimizer=optimizer,
    epoch=epoch,
    device='cpu'
  )

torch.save(fl, '../models/visual_contrastive.pth')
  

