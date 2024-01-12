import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim

# get dataloader for a specific modality and signal.
############################################################################################################
from clea.dataloaders.exploratory_loaders import RawChoiceDatasetwithTaskEmbedding
from clea.representation_models.train_model_utils import MultiEpochsDataLoader

def get_dataloader(batch_size: int, modality: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}"')
    dataset = RawChoiceDatasetwithTaskEmbedding(df, train=True, kind=modality, transform=torch.Tensor, data_dir='../data/')
    embedding_dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, num_workers=2, )

    return embedding_dataloader, dataset.get_input_dim()

############################################################################################################


# get model for a specific modality + training objective + signal
############################################################################################################
from clea.representation_models.auditory import RawAudioEncoder, RawAudioAE, RawAudioVAE
from clea.representation_models.visual import RawImageEncoder, RawImageAE, RawImageVAE
from clea.representation_models.kinetic import RawSequenceEncoder, Seq2Seq, Seq2SeqVAE

def get_model_and_loss_fn(model_type: str, 
                    modality: str,
                    input_dim: int = 1024,
                    hidden_dim: int = 512,
                    latent_dim: int = 64,
                    device: str = 'cpu'):
  
    if model_type == 'contrastive':
        if modality == 'auditory':
            return RawAudioEncoder(input_dim, hidden_dim, latent_dim, device=device), nn.TripletMarginLoss()
        elif modality == 'visual':
            return RawImageEncoder(input_dim, hidden_dim, latent_dim, device=device), nn.TripletMarginLoss()
        elif modality == 'kinesthetic':
            return RawSequenceEncoder(input_size=3, hidden_size=latent_dim, num_layers=2, device=device), nn.TripletMarginLoss()

    elif model_type == 'random':
        if modality == 'auditory':
            return RawAudioEncoder(input_dim, hidden_dim, latent_dim, device=device), None
        elif modality == 'visual':
            return RawImageEncoder(input_dim, hidden_dim, latent_dim, device=device), None
        elif modality == 'kinesthetic':
            return RawSequenceEncoder(input_size=3, hidden_size=latent_dim, num_layers=2, device=device), None

    elif model_type == 'autoencoder':
        if modality == 'auditory':
            return RawAudioAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'visual':
            return RawImageAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'kinesthetic':
            return Seq2Seq(input_size=3, hidden_size=latent_dim, num_layers=2, device=device), nn.MSELoss()
    
    elif model_type == 'contrastive+autoencoder':
        if modality == 'auditory':
            return RawAudioAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'visual':
            return RawImageAE(input_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'kinesthetic':
            return Seq2Seq(input_size=3, hidden_size=latent_dim, num_layers=2, device=device), nn.MSELoss()

    elif model_type == 'VAE':
        if modality == 'auditory':
            model = RawAudioVAE(input_dim, hidden_dim, latent_dim, device=device)
            return model, model.vae_loss
        elif modality == 'visual':
            model = RawImageVAE(input_dim, hidden_dim, latent_dim, device=device)
            return model, model.vae_loss
        elif modality == 'kinesthetic':
            model = Seq2SeqVAE(input_size=3, hidden_size=latent_dim, num_layers=2, device=device)
            return model, model.vae_loss


############################################################################################################


# train model for a specific loss
############################################################################################################
from clea.representation_models.train_model_utils import train_single_epoch_with_task_embedding
from clea.representation_models.pretrained import TaskEmbedder

if __name__ == '__main__':

    BATCH_SIZE = 32
    EMBEDDING_DIM = 64
    LR = 1e-4
    NUM_EPOCHS = 1
    DEVICE = 'mps'

    for modality in ['kinesthetic','auditory', 'visual']:
            for model_type in ['autoencoder', 'contrastive+autoencoder', 'contrastive', 'random']:
                        
                print(f'Training {modality} modality; {model_type} model;')

                #1. get dataloader
                data, input_dim = get_dataloader(batch_size=BATCH_SIZE, modality=modality)    

                #2. get model
                model, loss_fn = get_model_and_loss_fn(model_type=model_type, modality=modality, input_dim=input_dim, latent_dim=EMBEDDING_DIM, device=DEVICE)
                task_embedder = TaskEmbedder(EMBEDDING_DIM, DEVICE)

                #3. train model
                if loss_fn is not None:

                    optimizer = optim.Adam(model.parameters(), lr=LR)
                    embed_optimizer = optim.Adam(task_embedder.parameters(), lr=LR)
                    training_results = []
                    # set model to training mode on correct device
                    model.train()
                    model.to(DEVICE)
                    task_embedder.train()
                    task_embedder.to(DEVICE)

                    for i in tqdm(range(NUM_EPOCHS)):
                        iterations, avg_loss = train_single_epoch_with_task_embedding(embedding_type=model_type, model=model, 
                                                    task_embedder=task_embedder, loss_fn=loss_fn, data_loader=data, 
                                                    model_optimizer=optimizer, embed_optimizer=embed_optimizer, epoch=i, device=DEVICE)
                        training_results.append({'iters': iterations, 'loss': avg_loss})
                        tqdm.write(f'Epoch {i} loss: {avg_loss}')

                #4. save model and data
                torch.save(model, f'../data/trained_models/taskemb_{modality}_{model_type}_{EMBEDDING_DIM}.pth')
                pd.DataFrame(training_results).to_csv(f'../data/trained_models/taskemb_{modality}_{model_type}_{EMBEDDING_DIM}.csv')

############################################################################################################