from train.train_model_utils import get_dataloader, get_model, train_single_epoch

import torch
from torch import optim
from torch import nn
import pandas as pd
import matplotlib.pyplot as plt
#########################################
#                                       #
#         Experimental Parameters       #
#                                       #
#########################################

SIGNAL_MODALITY = 'visual' # must be one of 'visual', 'auditory', or 'kinesthetic'
EMBEDDING_TYPE = 'random' # must be one of 'contrastive+autoencoder', 'contrastive', 'autoencoder', or 'random'

embedding_size = 128
device = 'cpu' #  "cuda:0" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

#########################################
#                                       #
#           Training Procedure          #
#                                       #
#########################################
for SIGNAL_MODALITY in ['visual', 'auditory', 'kinesthetic']:
    for EMBEDDING_TYPE in ['VAE', 'contrastive+autoencoder', 'contrastive', 'autoencoder','random']:
        for signal in ['idle', 'searching', 'has_item', 'has_information']:
            dataset, input_dim = get_dataloader(SIGNAL_MODALITY, signal)
            model = get_model(EMBEDDING_TYPE, latent_dim=embedding_size, input_dim=input_dim, device=device)

            if EMBEDDING_TYPE != 'random':
                training_data = []
                optimizer = optim.Adam(model.parameters())

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
                data.to_csv(f'./trained_models/{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{signal}_{embedding_size}.csv')

                plt.plot(data['iterations'], data['avg_loss'])
                # plt.show()

            torch.save(model, f'./trained_models/{SIGNAL_MODALITY}_{EMBEDDING_TYPE}_{signal}_{embedding_size}.pth')
