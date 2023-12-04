from train.train_model_utils import get_dataloader, get_raw_data_model, train_single_epoch

from torch import optim
from torch import nn
import torch
import pandas as pd
import matplotlib.pyplot as plt

#########################################
#                                       #
#         Experimental Parameters       #
#                                       #
#########################################

SIGNAL_MODALITY = 'kinesthetic' # must be one of 'visual', 'auditory', or 'kinesthetic'
EMBEDDING_TYPE = 'VAE' # must be one of 'VAE', 'contrastive+autoencoder', 'contrastive', 'autoencoder', or 'random'

embedding_size = 128
device = "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device = 'cpu'
#########################################
#                                       #
#           Training Procedure          #
#                                       #
#########################################
for EMBEDDING_TYPE in ['random']:#'contrastive','contrastive+autoencoder', 'autoencoder','VAE', 'random']:
    for signal in ['idle', 'searching', 'has_item', 'has_information']:
        dataset, input_dim = get_dataloader(SIGNAL_MODALITY, signal, dataloader_type='raw_data')
        
        model = get_raw_data_model(EMBEDDING_TYPE, SIGNAL_MODALITY, latent_dim=embedding_size, input_dim=input_dim, device=device)

        if EMBEDDING_TYPE != 'random':
            training_data = []
            optimizer = optim.Adam(model.parameters(), lr=.002)

            if EMBEDDING_TYPE == 'contrastive':
                loss_fn = nn.TripletMarginLoss()
            elif EMBEDDING_TYPE == 'autoencoder' or EMBEDDING_TYPE == 'contrastive+autoencoder':
                loss_fn = nn.MSELoss()
            elif EMBEDDING_TYPE == 'VAE':
                loss_fn = model.vae_loss

            model.to(device)
            
            for epoch in range(300):
                iterations, avg_loss = train_single_epoch(EMBEDDING_TYPE, model, loss_fn, dataset, optimizer, epoch, device)
                training_data.append({
                    'iterations': iterations,
                    'avg_loss': avg_loss
                })
            
            data = pd.DataFrame(training_data)
            data.to_csv(f'./trained_models/raw_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{signal}_{embedding_size}.csv')

            plt.plot(data['iterations'], data['avg_loss'])
            # plt.show()

        torch.save(model, f'./trained_models/raw_{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{signal}_{embedding_size}.pth')

