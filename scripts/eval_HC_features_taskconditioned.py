import torch 
import torch.nn as nn
import torch.nn.functional as F
from clea.dataloaders import HCFeaturesTaskConditionedDataset
from clea.reward_models.model_definitions import HCFeatureLearner
from tqdm import tqdm
import pandas as pd

MODALITY = 'auditory'
EMBEDDING_TYPE = 'random'

results = []

for _ in tqdm(range(20)):
    for MODALITY in ['visual', 'auditory', 'kinesthetic']:
        for EMBEDDING_TYPE in ['random', 'autoencoder', 'contrastive+autoencoder']:
            dataset = HCFeaturesTaskConditionedDataset( f'../data/embeds/{MODALITY}_{EMBEDDING_TYPE}_64_taskconditioned.npy',
                                                        f'../data/handcrafted_features/{MODALITY}.npy', transform=torch.Tensor)


            input_size = dataset[0][0].shape[0]
            output_size = dataset[0][1].shape[0]

            train, test = torch.utils.data.random_split(dataset, [.8, .2])

            train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
            test_data_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)


            feature_predictor = HCFeatureLearner(input_size, 1024, output_size, device='cpu')
            if MODALITY == 'visual':
                loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
            elif MODALITY == 'auditory' or MODALITY == 'kinesthetic':
                loss_fn = nn.MSELoss(reduction='sum')

            optimizer = torch.optim.Adam(feature_predictor.parameters(), lr=1e-3)

            feature_predictor.train()

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
                'modality': MODALITY,
                'embedding_type': EMBEDDING_TYPE,
                'loss': total_loss / len(test_data_loader)
            })

pd.DataFrame(results).to_csv('../data/results/HC_feature_predictor_results.csv')



    
