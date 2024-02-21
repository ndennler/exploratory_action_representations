import torch
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
from torch import optim

# get dataloader for a specific modality and signal.
############################################################################################################
from clea.dataloaders.exploratory_loaders import RawChoiceDatasetwithTaskEmbedding
from clea.representation_models.train_model_utils import MultiEpochsDataLoader
from torch.utils.data import DataLoader

def get_dataloader(batch_size: int, modality: str, pretrained_embeds_path: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}"')
    dataset = RawChoiceDatasetwithTaskEmbedding(df, kind=modality, transform=torch.Tensor, data_dir='../data/', pretrained_embed_path=pretrained_embeds_path)
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

    return embedding_dataloader, dataset.get_input_dim(), dataset.get_output_dim()

############################################################################################################

# get model for a specific modality + training objective + signal
############################################################################################################
from clea.representation_models.auditory import  Pretrained2RawAudioAE, Pretrained2RawAudioVAE
from clea.representation_models.visual import Pretrained2RawImageAE, Pretrained2RawImageVAE
from clea.representation_models.kinetic import Pretrained2RawSeq2Seq, Pretrained2RawSeq2SeqVAE
from clea.representation_models.pretrained import PretrainedEncoder

def get_model_and_loss_fn(model_type: str, 
                    modality: str,
                    input_dim: int = 1024,
                    output_dim: list = [3,256,256],
                    hidden_dim: int = 512,
                    latent_dim: int = 64,
                    device: str = 'cpu'):
  
    if model_type == 'contrastive':
        return PretrainedEncoder(input_dim, hidden_dim, latent_dim, device=device), nn.TripletMarginLoss()

    elif model_type == 'random':
        return PretrainedEncoder(input_dim, hidden_dim, latent_dim, device=device), None

    elif model_type == 'autoencoder':
        if modality == 'auditory':
            return Pretrained2RawAudioAE(input_dim, output_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'visual':
            return Pretrained2RawImageAE(input_dim, output_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'kinesthetic':
            return Pretrained2RawSeq2Seq(input_dim, output_size = 3, hidden_size=hidden_dim, latent_dim=latent_dim, num_layers=2, device=device), nn.MSELoss()
    
    elif model_type == 'contrastive+autoencoder':
        if modality == 'auditory':
            return Pretrained2RawAudioAE(input_dim, output_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'visual':
            return Pretrained2RawImageAE(input_dim, output_dim, hidden_dim, latent_dim, device=device), nn.MSELoss()
        elif modality == 'kinesthetic':
            return Pretrained2RawSeq2Seq(input_dim, output_size = 3, hidden_size=hidden_dim, latent_dim=latent_dim, num_layers=2, device=device), nn.MSELoss()

    elif model_type == 'VAE':
        if modality == 'auditory':
            model = Pretrained2RawAudioVAE(input_dim, output_dim, hidden_dim, latent_dim, device=device)
            return model, model.vae_loss
        elif modality == 'visual':
            model = Pretrained2RawImageVAE(input_dim, output_dim, hidden_dim, latent_dim, device=device)
            return model, model.vae_loss
        elif modality == 'kinesthetic':
            model = Pretrained2RawSeq2SeqVAE(input_dim, output_size = 3, hidden_size=hidden_dim, latent_dim=latent_dim, num_layers=2, device=device)
            return model, model.vae_loss
        
############################################################################################################
        
# train model for a specific loss
############################################################################################################
from clea.representation_models.train_model_utils import train_single_epoch_with_task_embedding_from_pretrained
from clea.representation_models.pretrained import TaskEmbedder

if __name__ == '__main__':

    BATCH_SIZE = 128

    HIDDEN_DIM = 512
    EMBEDDING_DIM = 128

    LR = 1e-4
    NUM_EPOCHS = 300
    DEVICE = 'cuda:0'


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

        #1. get dataloader
        data, input_dim, output_dim = get_dataloader(batch_size=BATCH_SIZE, modality=modality, pretrained_embeds_path=em_path)    

        #2. get model
        model, loss_fn = get_model_and_loss_fn(model_type=model_type, modality=modality, input_dim=input_dim, output_dim=output_dim, hidden_dim=HIDDEN_DIM, latent_dim=EMBEDDING_DIM, device=DEVICE)
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
                iterations, avg_loss = train_single_epoch_with_task_embedding_from_pretrained(embedding_type=model_type, model=model, 
                                            task_embedder=task_embedder, loss_fn=loss_fn, data_loader=data, 
                                            model_optimizer=optimizer, embed_optimizer=embed_optimizer, epoch=i, device=DEVICE)
                training_results.append({'iters': iterations, 'loss': avg_loss})
                tqdm.write(f'Epoch {i} loss: {avg_loss}')

        #4. save model and data
        torch.save(model, f'../data/trained_models/{modality}&taskconditioned&{embed_name}&{model_type}&all_signals&{EMBEDDING_DIM}.pth')
        torch.save(task_embedder, f'../data/trained_models/{modality}&taskconditioned&{embed_name}&{model_type}&all_signals&{EMBEDDING_DIM}_embedder.pth')

        pd.DataFrame(training_results).to_csv(f'../data/trained_models/{modality}&taskconditioned&{embed_name}&{model_type}&all_signals&{EMBEDDING_DIM}.csv')

############################################################################################################