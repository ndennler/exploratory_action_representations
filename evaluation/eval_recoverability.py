import os
import torch

import pandas as pd
import numpy as np

from clea.dataloaders import UserStudyQueryDataloader
from clea.reward_models import RewardLearner
from torch.utils.data import DataLoader
from torchmetrics import Accuracy

device = 'cpu'
results = []
MODALITY= 'kinetic'

for f in os.listdir('../data/representation_evaluation_data'):
    pid, modality, signal = f[:-4].split('&')
    if modality != MODALITY:
         continue

    data = np.load(f'../data/representation_evaluation_data/{f}')

    embeds = np.load(f'../data/embeds/kinesthetic&independent&raw&contrastive&all_signals&128.npy')

    train_loader = DataLoader(UserStudyQueryDataloader(data['train'], embeds, signal, transform=torch.Tensor), 16)

    reward_model = RewardLearner(128)
    reward_model.to(device)
    optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

    #training loop
    for _ in range(15):
        for option1, option2, choice in train_loader:
            optimizer.zero_grad()

            r1 = reward_model(option1.to(device))
            r2 = reward_model(option2.to(device))
            
            rewards = torch.cat((r1,r2), 1)

            loss = torch.nn.CrossEntropyLoss()(rewards, choice.to(device))
            loss.backward()
            optimizer.step()
            # print(loss.item())
    
    #evaluation loop
    reward_model.eval()
    eval_loader = DataLoader(UserStudyQueryDataloader(data['test'], embeds, signal, transform=torch.Tensor), 16)

    eval_fn = Accuracy(task="multiclass", num_classes=2).to(device)
    eval_values = []

    for option1, option2, choice in train_loader:
            optimizer.zero_grad()

            r1 = reward_model(option1.to(device))
            r2 = reward_model(option2.to(device))
            rewards = torch.cat((r1,r2), 1)

            eval_result = eval_fn(rewards, choice)
            eval_values.append(eval_result.item())

    print('Accuracy:')
    print(np.nanmean(eval_values))
    results.append(np.nanmean(eval_values))

print(np.mean(results))