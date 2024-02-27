import os
import pandas as pd
import torch
import torch.nn as nn

from clea.dataloaders import HCFeaturesTaskConditionedDataset
from clea.reward_models.model_definitions import HCFeatureLearner

from tqdm import tqdm

experiments = os.listdir('../data/embeds')
results = []

for i, experiment in enumerate(experiments):
    print(f'Running experiment {i+1}/{len(experiments)}')
    modality, conditioning, pretraining, embedding_type, signals, size = experiment[:-4].split('&')
    if size != '128':
        continue
    
    for _ in range(2):
        for task_idx in range(4):
            dataset = HCFeaturesTaskConditionedDataset( f'../data/embeds/{experiment}',
                                                        f'../data/handcrafted_features/{modality}.npy', 
                                                        transform=torch.Tensor,
                                                        task_idx = task_idx)
            
            input_size = dataset[0][0].shape[0]
            output_size = dataset[0][1].shape[0]

            train, test = torch.utils.data.random_split(dataset, [.8, .2])
            train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
            test_data_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)

            feature_predictor = HCFeatureLearner(input_size, 1024, output_size, device='cpu')
            feature_predictor.train()
            loss_fn = nn.MSELoss(reduction='sum')
            optimizer = torch.optim.Adam(feature_predictor.parameters(), lr=1e-3)

            for _ in tqdm(range(7)):
                for batch_idx, (embedding, hc_features) in enumerate(train_data_loader):
                    optimizer.zero_grad()

                    pred = feature_predictor(embedding)    

                    loss = loss_fn(pred, hc_features.float())
                    loss.backward()
                    optimizer.step()

            total_loss = 0
            for batch_idx, (embedding, hc_features) in enumerate(test_data_loader):
                pred = feature_predictor(embedding)
                loss = loss_fn(pred, hc_features)
                total_loss += loss.item()

            print(f'Average Loss : {total_loss / len(test_data_loader)}')

            results.append({
                'modality': modality,
                'conditioning': conditioning,
                'pretraining': pretraining,
                'embedding_type': embedding_type,
                'embedding_size': size,
                'task_idx': task_idx, 
                'loss': total_loss / len(test_data_loader)
            })

pd.DataFrame(results).to_csv('../data/results/FPE_results.csv')