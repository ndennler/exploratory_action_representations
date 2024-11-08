import os
import torch

import pandas as pd
import numpy as np

from clea.dataloaders import UserStudyQueryDataloader
from clea.reward_models import RewardLearner
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from sklearn.decomposition import PCA
from umap import UMAP
from sklearn.random_projection import GaussianRandomProjection

device = 'cpu'
results = []


for embed_fname in ['../data/visual/xclip_embeds.npy','../data/auditory/AST_embeds.npy', '../data/kinetic/xclip_embeds.npy']:
    embed_modality = embed_fname.split('/')[2]
    method = 'pretrained'
    # em_size = 512 if embed_modality in ['visual','kinetic'] else 768
    em_size = 128

    # embed_modality, _, _, method, _ , em_size = embed_fname[:-4].split('&')
    embeds = np.load(f'{embed_fname}')
    reduced_array = embeds
    # Apply PCA to reduce to (5912, 64)
    # pca = PCA(n_components=em_size)
    # reduced_array = pca.fit_transform(embeds)

    # umap = UMAP(n_components=em_size, n_neighbors=15, min_dist=0.5)
    # reduced_array = umap.fit_transform(embeds)

    rp = GaussianRandomProjection(n_components=em_size)
    reduced_array = rp.fit_transform(embeds)
    # reduced_array = torch.sigmoid(torch.tensor(reduced_array)).numpy()

    # Tile to get shape (5912, 4, 64)
    embeds = np.tile(reduced_array[:, np.newaxis, :], (1, 4, 1))


    
    
    print(embed_modality, method, em_size)

    #TODO: remove this line when doing the full experiment
    # if em_size != '128':
    #     continue

    for participant_fname in os.listdir('../data/representation_evaluation_data'):
        # print(participant_fname)
        if '.npz' not in participant_fname:
            continue
        pid, modality, signal = participant_fname[:-4].split('&')

        if modality != embed_modality:
            if not (modality == 'kinetic' and embed_modality == 'kinesthetic'):
                continue

        data = np.load(f'../data/representation_evaluation_data/{participant_fname}')
        

        train_loader = DataLoader(UserStudyQueryDataloader(data['train'], embeds, signal, transform=torch.Tensor), 32)

        reward_model = RewardLearner(int(em_size), hidden_dim=256)
        reward_model.to(device)
        optimizer = torch.optim.Adam(reward_model.parameters(), lr=1e-3)

        #training loop
        for iter in range(60):
            for option1, option2, choice in train_loader:
                optimizer.zero_grad()

                r1 = reward_model(option1.to(device))
                r2 = reward_model(option2.to(device))
                
                rewards = torch.cat((r1,r2), 1)

                loss = torch.nn.CrossEntropyLoss()(rewards, choice.to(device)) + 0.01*torch.norm(r1, p=1) + 0.01*torch.norm(r2, p=1)
                loss.backward()
                optimizer.step()
                # print(iter, loss.item())
        
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

df = pd.DataFrame(results)
df.to_csv('nn_pretrained_results.csv')

print(df.groupby(['modality']).mean())