from train.train_model_utils import get_dataloader, get_raw_data_model, train_single_epoch_with_task_embedding
from train.model_definitions import TaskEmbedder
import torch
from torch import optim
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

SIGNAL_MODALITY = 'kinesthetic' # must be one of 'visual', 'auditory', or 'kinesthetic'
EMBEDDING_TYPE = 'contrastive+autoencoder' # must be one of 'VAE', 'contrastive+autoencoder', 'contrastive', 'autoencoder', or 'random'

embedding_size = 128
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
######################################### 
#                                       #
#           Training Procedure          #
#                                       #
#########################################
for SIGNAL_MODALITY in ['visual', 'auditory']:
    for EMBEDDING_TYPE in ['contrastive+autoencoder', 'contrastive', 'autoencoder', 'random']:

        if SIGNAL_MODALITY == 'kinesthetic':
            device = 'cpu'
        else:
            device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

        dataset, input_dim = get_dataloader(SIGNAL_MODALITY, 'all_signals', dataloader_type='task_embedding')
        print(dataset.dataset.task_to_index_mapping)
        model = get_raw_data_model(EMBEDDING_TYPE, SIGNAL_MODALITY, latent_dim=embedding_size, input_dim=input_dim, device=device)
        task_embedder = TaskEmbedder(embedding_size, device)

        if EMBEDDING_TYPE != 'random':
            training_data = []
            model_optimizer = optim.Adam(model.parameters(), lr=.002)
            embedding_optimizer = optim.Adam(task_embedder.parameters(), lr=.002)

            if EMBEDDING_TYPE == 'contrastive':
                loss_fn = nn.TripletMarginLoss()
            elif EMBEDDING_TYPE == 'autoencoder' or EMBEDDING_TYPE == 'contrastive+autoencoder':
                loss_fn = nn.MSELoss()
            elif EMBEDDING_TYPE == 'VAE':
                loss_fn = model.vae_loss

            model.to(device)
            task_embedder.to(device)
            
            for epoch in tqdm(range(300)):
                iterations, avg_loss = train_single_epoch_with_task_embedding(EMBEDDING_TYPE, model, task_embedder, loss_fn, dataset, model_optimizer, embedding_optimizer, epoch, device)
                training_data.append({
                    'iterations': iterations,
                    'avg_loss': avg_loss
                })
            
            data = pd.DataFrame(training_data)
            data.to_csv(f'./trained_models/raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}.csv')

            plt.plot(data['iterations'], data['avg_loss'])

        torch.save(model, f'./trained_models/raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}.pth')
        torch.save(task_embedder, f'./trained_models/raw_taskembedding_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{embedding_size}_task_embedder.pth')