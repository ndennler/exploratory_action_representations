import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim

# get dataloader for a specific modality and signal.
############################################################################################################
from clea.dataloaders.exploratory_loaders import ChoiceDataset
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int, modality: str, signal: str, embed_path: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}" & signal == "{signal}"')
    dataset = ChoiceDataset(df, embed_path=embed_path, train=True, kind=modality, transform=torch.Tensor)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

    return embedding_dataloader, dataset.get_input_dim()

############################################################################################################


# get model for a specific modality + training objective + signal
############################################################################################################
from clea.representation_models.pretrained import PretrainedEncoder, AEPretrainedLearner, VAEPretrainedLearner

def get_model_and_loss_fn(model_type: str, 
              input_dim: int = 1024,
              hidden_dim: int = 512,
              latent_dim: int = 64,
              device: str = 'cpu'):
  
    if model_type == 'contrastive':
        return PretrainedEncoder(input_dim, hidden_dim, latent_dim, device=device), nn.TripletMarginLoss()

    elif model_type == 'random':
        return PretrainedEncoder(input_dim, hidden_dim, latent_dim, device=device), None

    elif model_type == 'autoencoder':
        return AEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
    
    elif model_type == 'contrastive+autoencoder':
        return AEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()

    elif model_type == 'VAE':
        model = VAEPretrainedLearner(input_dim, hidden_dim, latent_dim, device=device)
        return model, model.vae_loss

############################################################################################################


# train model for a specific loss
############################################################################################################
from clea.representation_models.train_model_utils import train_single_epoch

if __name__ == '__main__':

    BATCH_SIZE = 32
    EMBEDDING_DIM = 64
    LR = 1e-4
    NUM_EPOCHS = 3
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

    for num, e in enumerate(experiments):
        print(f'STARTING EXPERIMENT {num+1} OF {len(experiments)}')

        modality, model_type, em_path = e['modality'], e['model_type'], e['pretrained_embeds_path']
        embed_name = em_path.split('/')[-1].split('.')[0]

        for signal in ['idle', 'searching', 'has_item', 'has_information']:                

            #1. get dataloader
            data, input_dim = get_dataloader(batch_size=BATCH_SIZE, modality=modality, signal=signal, embed_path=em_path)    

            #2. get model
            model, loss_fn = get_model_and_loss_fn(input_dim=input_dim, model_type=model_type, latent_dim=EMBEDDING_DIM, device=DEVICE)

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
            torch.save(model, f'../data/trained_models/{modality}&independent&{embed_name}&{model_type}&{signal}&{EMBEDDING_DIM}.pth')
            pd.DataFrame(training_results).to_csv(f'../data/trained_models/{modality}&independent&{embed_name}&{model_type}&{signal}&{EMBEDDING_DIM}.csv')

    ############################################################################################################