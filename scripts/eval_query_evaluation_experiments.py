import torch
from torch import nn
from torch import optim
from torchmetrics import Accuracy

import pandas as pd
import numpy as np
import os

from train.model_definitions import ContrastiveAEPretrainedLearner,ContrastivePretrainedLearner, AEPretrainedLearner, RandomPretrainedLearner, VAEPretrainedLearner
from evaluation.model_definitions import RewardLearner
from evaluation.eval_utils import get_pids_for_training, get_train_test_dataloaders, train_single_epoch, eval_model, calc_reward, generate_embeddings

#########################################
#                                       #
#         Experimental Parameters       #
#                                       #
#########################################

SIGNAL_MODALITY = 'kinesthetic' # must be one of 'visual', 'auditory', or 'kinesthetic'
EMBEDDING_TYPE =  'contrastive+autoencoder' # must be one of 'contrastive+autoencoder', 'contrastive', 'autoencoder', or 'random'


CALC_REWARD = False # true if you only want to calculate the final reward
FROM_RAW_DATA = False

embedding_size = 128 # has to be the same as the training embedding size
seed=13
#########################################
#                                       #
#         Evaluation Procedure          #
#                                       #
#########################################
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cuda:0'
# 'visual', 
for SIGNAL_MODALITY in ['visual', 'auditory', 'kinesthetic']:
    for EMBEDDING_TYPE in ['contrastive+autoencoder', 'contrastive','autoencoder','VAE', 'random']:
        for signal in ['idle', 'searching', 'has_item', 'has_information']:

            model_name = f'{"raw_" if FROM_RAW_DATA else ""}{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{signal}_{embedding_size}.pth'
            embedding_model = torch.load('../trained_models/' + model_name)
            embedding_model.eval()
            embedding_model.to(device)

            if FROM_RAW_DATA:
                generate_embeddings(embedding_model, model_name, SIGNAL_MODALITY, embedding_size, device)

            results = []

            for pid in get_pids_for_training(SIGNAL_MODALITY, signal):
                
                if CALC_REWARD:
                    train_data = get_train_test_dataloaders(PID=pid, signal_modality=SIGNAL_MODALITY, signal=signal, train_only=True, raw_data=FROM_RAW_DATA)
                    print(pid, len(train_data), signal)
                    final_data = pd.read_csv('./data/evaluation/concatenated_final_signals.csv')
                    final_data = final_data.query(f'pid=={pid} & signal=="{signal}"')[SIGNAL_MODALITY].values[0]
                else:
                    training_sets, testing_sets = get_train_test_dataloaders(pid, SIGNAL_MODALITY, signal, raw_data=FROM_RAW_DATA, model_name=model_name)
                
                
                for train_data, test_data in zip(training_sets, testing_sets):

                    reward_model = RewardLearner(embedding_size)

                    for epoch in range(30):
                        train_single_epoch(
                            embedding_model=embedding_model,
                            reward_model=reward_model,
                            loss_fn= nn.CrossEntropyLoss(),
                            data_loader=train_data,
                            optimizer=optim.Adam(reward_model.parameters()),
                            epoch=epoch,
                            device=device,
                        )
                    
                    if CALC_REWARD:
                        reward = calc_reward(
                                embedding_model=embedding_model,
                                reward_model=reward_model,
                                data_loader=train_data,
                                index=final_data,
                                device=device
                            )
                        results.append(reward)
                    else:
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
                        'signal': signal,
                        'embedding_type': EMBEDDING_TYPE,
                        'embedding_size': embedding_size,
                        'metric': res,
                        'seed': seed
                    })

            if os.path.exists('./data/results.csv'):
                df = pd.read_csv('./data/results.csv')
                pd.concat([df, pd.DataFrame(data)]).to_csv('../data/results.csv', index=False)
            else:
                pd.DataFrame(data).to_csv('../data/results.csv', index=False)



    
