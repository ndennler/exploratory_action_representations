import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy
from tqdm import tqdm

import pandas as pd
import numpy as np
import os

# from train.model_definitions import ContrastiveAEPretrainedLearner,ContrastivePretrainedLearner, AEPretrainedLearner, RandomPretrainedLearner, VAEPretrainedLearner
from clea.reward_models.model_definitions import RewardLearner
from clea.reward_models.eval_utils import get_pids_for_training, get_train_test_dataloaders, train_single_epoch_task_embeds, eval_model


EMBEDDING_DIM = 64 # has to be the same as the training embedding size
CALC_REWARD = False
FROM_RAW_DATA = True
seed = 42

DEVICE = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
DEVICE = 'cpu'

experiments = [
    {'modality': 'visual', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'random', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},

    {'modality': 'visual', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'random', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},

    {'modality': 'auditory', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    {'modality': 'auditory', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    {'modality': 'auditory', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    {'modality': 'auditory', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    {'modality': 'auditory', 'model_type': 'random', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    
    {'modality': 'auditory', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    {'modality': 'auditory', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    {'modality': 'auditory', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    {'modality': 'auditory', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    {'modality': 'auditory', 'model_type': 'random', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},

    {'modality': 'kinesthetic', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    {'modality': 'kinesthetic', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    {'modality': 'kinesthetic', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    {'modality': 'kinesthetic', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    {'modality': 'kinesthetic', 'model_type': 'random', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
]





if __name__ == '__main__':
        
    for num, e in enumerate(experiments):
        print(f'EVALUATING QUERIES FOR EXPERIMENT {num+1} OF {len(experiments)}')
        modality, model_type, em_path = e['modality'], e['model_type'], e['pretrained_embeds_path']
        embed_name = em_path.split('/')[-1].split('.')[0]

        # model_name = f'raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}.pth'
        # embedding_model = torch.load('../final_models/' + model_name)

        # model_name = f'raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}_task_embedder.pth'
        # task_embedder = torch.load('./final_models/' + model_name)

        embedding_model = torch.load(f'../data/final_models/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}.pth', map_location=torch.device(DEVICE))
        task_embedder = torch.load(f'../data/final_models/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}_embedder.pth', map_location=torch.device(DEVICE))

        embedding_model.eval()
        embedding_model.to(DEVICE)
        task_embedder.eval()
        task_embedder.to(DEVICE)

        # generate_embeddings_task_conditioned(embedding_model, task_embedder, model_name, SIGNAL_MODALITY, embedding_size, device)

        results = []

        for pid in tqdm(get_pids_for_training(modality, 'all_signals')):
            training_sets, testing_sets = get_train_test_dataloaders(pid, modality, 'all_signals', raw_data=FROM_RAW_DATA, model_name=f'taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}')
            # print(training_sets, testing_sets)
            
            for train_data, test_data in zip(training_sets, testing_sets):

                reward_model = RewardLearner(EMBEDDING_DIM)

                for epoch in range(30):
                    train_single_epoch_task_embeds(
                        embedding_model=embedding_model,
                        task_embedder=task_embedder,
                        reward_model=reward_model,
                        loss_fn= nn.CrossEntropyLoss(),
                        data_loader=train_data,
                        optimizer=optim.Adam(reward_model.parameters()),
                        device=DEVICE,
                    )
                
                acc = eval_model(
                        embedding_model=embedding_model,
                        reward_model=reward_model,
                        eval_fn=Accuracy(task="multiclass", num_classes=4).to(DEVICE),
                        # eval_fn=nn.CrossEntropyLoss(),
                        data_loader=test_data,
                        device=DEVICE
                    )
                
                # print(test_data.dataset.data.iloc[0]['signal'])             
                results.append([acc, test_data.dataset.data.iloc[0]['signal']])



        data = []
        for res in results:
            if not np.isnan(res[0]):
                data.append({
                    'modality': modality,
                    # 'signal': signal,
                    'embed_name': embed_name,
                    'embedding_type': model_type,
                    'embedding_size': EMBEDDING_DIM,
                    'metric': res[0],
                    'signal': res[1],
                    'seed': seed
                })

        if os.path.exists('../data/results/taskembedding_TPA_results.csv'):
            df = pd.read_csv('../data/results/taskembedding_TPA_results.csv')
            pd.concat([df, pd.DataFrame(data)]).to_csv('../data/results/taskembedding_TPA_results.csv', index=False)
        else:
            pd.DataFrame(data).to_csv('../data/results/taskembedding_TPA_results.csv', index=False)
