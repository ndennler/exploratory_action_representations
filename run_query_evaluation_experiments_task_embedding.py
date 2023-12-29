import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy

import pandas as pd
import numpy as np
import os

from train.model_definitions import ContrastiveAEPretrainedLearner,ContrastivePretrainedLearner, AEPretrainedLearner, RandomPretrainedLearner, VAEPretrainedLearner
from evaluation.model_definitions import RewardLearner
from evaluation.eval_utils import get_pids_for_training, get_train_test_dataloaders, train_single_epoch_task_embeds, eval_model, calc_reward, generate_embeddings_task_conditioned


embedding_size = 128 # has to be the same as the training embedding size
CALC_REWARD = False
FROM_RAW_DATA = True
seed = 42

device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'

for SIGNAL_MODALITY in ['kinesthetic','visual', 'auditory']:
    for EMBEDDING_TYPE in ['contrastive+autoencoder', 'contrastive','autoencoder', 'random']:

            model_name = f'raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}.pth'
            embedding_model = torch.load('./trained_models/' + model_name)

            model_name = f'raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}_task_embedder.pth'
            task_embedder = torch.load('./trained_models/' + model_name)

            embedding_model.eval()
            embedding_model.to(device)
            task_embedder.eval()
            task_embedder.to(device)

            generate_embeddings_task_conditioned(embedding_model, task_embedder, model_name, SIGNAL_MODALITY, embedding_size, device)

            results = []

            for pid in get_pids_for_training(SIGNAL_MODALITY, 'all_signals'):
                training_sets, testing_sets = get_train_test_dataloaders(pid, SIGNAL_MODALITY, 'all_signals', raw_data=FROM_RAW_DATA, model_name=model_name)
                print(training_sets, testing_sets)
                
                for train_data, test_data in zip(training_sets, testing_sets):

                    reward_model = RewardLearner(embedding_size)

                    for epoch in range(30):
                        train_single_epoch_task_embeds(
                            embedding_model=embedding_model,
                            task_embedder=task_embedder,
                            reward_model=reward_model,
                            loss_fn= nn.CrossEntropyLoss(),
                            data_loader=train_data,
                            optimizer=optim.Adam(reward_model.parameters()),
                            epoch=epoch,
                            device=device,
                        )
                    
                    acc = eval_model(
                            embedding_model=embedding_model,
                            reward_model=reward_model,
                            eval_fn=Accuracy(task="multiclass", num_classes=4).to(device),
                            # eval_fn=nn.CrossEntropyLoss(),
                            data_loader=test_data,
                            device=device
                        )
                    results.append(acc)



            data = []
            for res in results:
                if not np.isnan(res):
                    data.append({
                        'modality': SIGNAL_MODALITY,
                        # 'signal': signal,
                        'embedding_type': EMBEDDING_TYPE,
                        'embedding_size': embedding_size,
                        'metric': res,
                        'seed': seed
                    })

            if os.path.exists('./data/task_embedding_results.csv'):
                df = pd.read_csv('./data/task_embedding_results.csv')
                pd.concat([df, pd.DataFrame(data)]).to_csv('./data/task_embedding_results.csv', index=False)
            else:
                pd.DataFrame(data).to_csv('./data/task_embedding_results.csv', index=False)
