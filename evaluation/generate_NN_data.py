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


for embed_fname in os.listdir('../data/embeds'):
    embed_modality, _, _, method, _ , em_size = embed_fname[:-4].split('&')
    print(embed_modality, method, em_size)

    #TODO: remove this line when doing the full experiment
    if em_size != '128':
        continue

    for participant_fname in os.listdir('../data/representation_evaluation_data'):
        pid, modality, signal = participant_fname[:-4].split('&')

        if modality != embed_modality:
            if not (modality == 'kinetic' and embed_modality == 'kinesthetic'):
                continue

        data = np.load(f'../data/representation_evaluation_data/{participant_fname}')
        embeds = np.load(f'../data/embeds/{embed_fname}')

        train_loader = DataLoader(UserStudyQueryDataloader(data['train'], embeds, signal, transform=torch.Tensor), 16)

        reward_model = RewardLearner(int(em_size), hidden_dim=256)
        reward_model.to(device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

        #training loop
        for _ in range(60):
            for option1, option2, choice in train_loader:
                optimizer.zero_grad()

                r1 = reward_model(option1.to(device))
                r2 = reward_model(option2.to(device))
                
                rewards = torch.cat((r1,r2), 1)

                loss = torch.nn.CrossEntropyLoss()(rewards, choice.to(device)) + 0.01*torch.norm(r1, p=1) + 0.01*torch.norm(r2, p=1)
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

        #log experimental results
        results.append({
             'pid': pid,
             'modality': modality,
             'method': method,
             'embedding_size': em_size,
             'accuracy': np.nanmean(eval_values),
        })

pd.DataFrame(results).to_csv('nn_results.csv')