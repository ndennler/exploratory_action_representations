import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy
from tqdm import tqdm

import pandas as pd
import os

from clea.reward_models.model_definitions import RewardLearner
from clea.reward_models.eval_utils import get_dataloaders, train_reward_model_one_epoch, evaluate_reward_model

#initialize variables
query_df = pd.read_csv('../data/evaluation/all_queries.csv')
PIDS = query_df['pid'].unique()
DEVICE = 'cpu'
results = []
experiments = os.listdir('../data/embeds')

for i, experiment in enumerate(experiments):
    print(f'Running experiment {i+1}/{len(experiments)}')
    if '128' not in experiment:
         continue
    
    modality, conditioning, pretraining, embedding_type, signals, size = experiment[:-4].split('&')
    
    # get query data loaders
    for pid in tqdm(PIDS):
        train_sets, test_sets = get_dataloaders(PID=pid, modality=modality, conditioning=conditioning, embedding_path=experiment[:-4])

        #train and eval reward model for each split
        for train, test in zip(train_sets, test_sets):

            reward_model = RewardLearner(int(size))

            for epoch in range(10):
                    loss = train_reward_model_one_epoch(
                        reward_model=reward_model,
                        loss_fn= nn.CrossEntropyLoss(),
                        data_loader=train,
                        optimizer=optim.Adam(reward_model.parameters()),
                        device=DEVICE,
                    )
                
            acc = evaluate_reward_model(
                    reward_model=reward_model,
                    eval_fn=Accuracy(task="multiclass", num_classes=4).to(DEVICE),
                    # eval_fn=nn.CrossEntropyLoss(),
                    data_loader=test,
                    device=DEVICE
                )
                     
            results.append({
                    'modality': modality,
                    'conditioning': conditioning,
                    'pretraining': pretraining,
                    'embedding_type': embedding_type,
                    'embedding_size': size,
                    'metric': acc,
                    'signal': test.dataset.data.iloc[0]['signal'],
                 })
        
            
if os.path.exists('../data/results/TPA_results.csv'):
    df = pd.read_csv('../data/results/TPA_results.csv')
    pd.concat([df, pd.DataFrame(results)]).to_csv('../data/results/TPA_results.csv', index=False)
else:
    pd.DataFrame(results).to_csv('../data/results/TPA_results.csv', index=False)
