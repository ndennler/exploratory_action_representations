import torch
import os
import pandas as pd
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from clea.reward_models.eval_utils import generate_all_embeddings_taskconditioned, generate_all_embeddings_independent

# get dataloader for a specific modality and signal.
############################################################################################################
from clea.dataloaders.exploratory_loaders import RawChoiceDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int, modality: str, signal: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}" & signal == "{signal}"')
    dataset = RawChoiceDataset(df, kind=modality, transform=torch.Tensor, data_dir='../data/')
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

    return embedding_dataloader, dataset.get_input_dim()

# set up models
############################################################################################################
from clea.representation_models.visual import RawImageEncoder, RawImageAE, RawImageVAE
def get_model_and_loss_fn(model_type: str, 
                    input_dim: int = 1024,
                    hidden_dim: int = 512,
                    latent_dim: int = 64,
                    parameter: float = 1,
                    device: str = 'cpu'):
  
    if model_type == 'contrastive':
            return RawImageEncoder(input_dim, hidden_dim, latent_dim, device=device), nn.TripletMarginLoss(margin=parameter)
    elif model_type == 'random':
            return RawImageEncoder(input_dim, hidden_dim, latent_dim, device=device), None
    elif model_type == 'autoencoder':
            return RawImageAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
    elif model_type == 'contrastive+autoencoder':
            return RawImageAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
    elif model_type == 'VAE':
            model = RawImageVAE(input_dim, hidden_dim, latent_dim, device=device, beta=parameter)
            return model, model.vae_loss
    elif model_type == 'contrastive+VAE':
            model = RawImageVAE(input_dim, hidden_dim, latent_dim, device=device)
            return model, model.vae_loss

# generate embeddings
############################################################################################################
def generate_independent_embeddings(model_path, DEVICE):
    modality, embedding_type, signal, size = model_path[:-4].split('&')
    model = torch.load(f'../data/trained_models2/{model_path}', map_location=torch.device(DEVICE))

    model.eval()
    model.device = DEVICE
    model.to(DEVICE)

    mode_to_type = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
    df = pd.read_csv('../data/all_data.csv').query(f'type=="{mode_to_type[modality]}"')

    if os.path.exists(f'../data/embeds/{modality}&{embedding_type}&all_signals&{size}.npy'):
        embed_storage_path = f'../data/embeds/{modality}&{embedding_type}&all_signals&{size}.npy'
    else:
        embed_storage_path = None

    embeds = generate_all_embeddings_independent(model=model, dataframe=df, embedding_size=int(size), signal=signal, 
                        device=DEVICE, data_dir='../data', pretrained_embeds_array=None, embed_storage_path=embed_storage_path)


    np.save(f'../data/embeds/{modality}&{embedding_type}&all_signals&{size}.npy', embeds)

# train model for a specific loss
############################################################################################################
from clea.representation_models.train_model_utils import train_single_epoch
def train(model_type, DEVICE='cpu ', BATCH_SIZE=128, EMBEDDING_DIM=128, LR=1e-4, NUM_EPOCHS=2, parameter=1):
        '''
        model type one of: 'contrastive', 'autoencoder', 'VAE'
        '''
        for signal in ['idle', 'searching', 'has_item', 'has_information']:
                print(f'Training visual modality;  {signal} signal; {model_type} model;')

                #1. get dataloader
                data, input_dim = get_dataloader(batch_size=BATCH_SIZE, modality='visual', signal=signal)    

                #2. get model
                model, loss_fn = get_model_and_loss_fn(model_type=model_type, input_dim=input_dim, latent_dim=EMBEDDING_DIM, device=DEVICE, parameter=parameter)

                #3. train model
                if loss_fn is not None:
                        optimizer = optim.Adam(model.parameters(), lr=LR)
                        training_results = []
                        # set model to training mode on correct device
                        model.train()
                        model.to(DEVICE)

                        for i in tqdm(range(NUM_EPOCHS)):
                                iterations, avg_loss = train_single_epoch(embedding_type=model_type, model=model, 
                                                        loss_fn=loss_fn, data_loader=data, optimizer=optimizer, epoch=i, device=DEVICE)
                                training_results.append({'iters': iterations, 'loss': avg_loss})
                                tqdm.write(f'Epoch {i} loss: {avg_loss}')

                #4. save model and data
                torch.save(model, f'../data/trained_models2/visual&{model_type}&{signal}&{EMBEDDING_DIM}.pth')

                # generate embeddings
                # generate_independent_embeddings(f'visual&{model_type}&{signal}&{EMBEDDING_DIM}.pth', DEVICE)

# eval model
############################################################################################################
from clea.dataloaders.query_loaders import RawQueryDataset
from clea.reward_models.eval_utils import eval_model
import clea.reward_models.eval_utils
from clea.reward_models.model_definitions import RewardLearner
from torchmetrics import Accuracy

def eval(model_type, EMBEDDING_DIM=128, DEVICE='cpu'):
        query_df = pd.read_csv('../data/evaluation/all_queries.csv')
        results = []

        for sig in ['idle', 'searching', 'has_information', 'has_item']:
                embedding_model = torch.load(f'../data/trained_models2/visual&{model_type}&{sig}&{EMBEDDING_DIM}.pth', map_location=torch.device(DEVICE))
               
                for PID in query_df['pid'].unique():
                        query = f'type == "visual" and pid == {PID}'

                        df = query_df.query(f'{query} and signal == "{sig}"')
                        if len(df) < 2:
                                continue
                        train_idx = len(df) // 2

                        train_df = df.iloc[train_idx:]
                        test_df = df.iloc[:train_idx]
                        
                        train_data = RawQueryDataset(train_df, kind='visual', transform=torch.Tensor, data_dir='../data/')
                        train_dataloader = DataLoader(train_data, batch_size=32, shuffle=True)

                        
                        reward_model = RewardLearner(EMBEDDING_DIM)

                        for i in range(10):
                                clea.reward_models.eval_utils.train_single_epoch(embedding_model=embedding_model, 
                                                reward_model=reward_model, 
                                                data_loader=train_dataloader, 
                                                loss_fn=nn.CrossEntropyLoss(), 
                                                optimizer=optim.Adam(reward_model.parameters()),
                                                epoch=i, 
                                                device=DEVICE)
                        
                        test_data = RawQueryDataset(test_df, kind='visual', transform=torch.Tensor, data_dir='../data/')
                        test_dataloader = DataLoader(test_data, batch_size=32)
                        acc = eval_model(
                                embedding_model=embedding_model,
                                reward_model=reward_model,
                                eval_fn=Accuracy(task="multiclass", num_classes=4).to(DEVICE),
                                data_loader=test_dataloader,
                                device=DEVICE
                        )

                        results.append(acc)

        print(f'Average accuracy: {np.mean(results)}')
        return np.nanmean(results)



if __name__ == '__main__':

        BATCH_SIZE = 64
        EMBEDDING_DIM = 128
        LR = 1e-3
        NUM_EPOCHS = 50
        DEVICE = 'cpu'
        
        #change these test values; train to get ideal value for contrastive loss and VAE loss
        # for parameter in [0.1, 0.5, .9, 2, 5, 10]:
        for parameter in [.1]:
                train('contrastive', DEVICE, BATCH_SIZE, EMBEDDING_DIM, LR, NUM_EPOCHS, parameter=parameter)
                eval('contrastive', EMBEDDING_DIM, DEVICE)

    

############################################################################################################
