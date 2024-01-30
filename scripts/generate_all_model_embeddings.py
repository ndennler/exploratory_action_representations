import torch
import pandas as pd
import numpy as np
from clea.reward_models.eval_utils import generate_all_embeddings_taskconditioned


EMBEDDING_DIM = 64
TASK_CONDITIONED = True
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
    print(f'GENERATING EMBEDDINGS FOR EXPERIMENT {num+1} OF {len(experiments)}')
    modality, model_type, em_path = e['modality'], e['model_type'], e['pretrained_embeds_path']
    embed_name = em_path.split('/')[-1].split('.')[0]

    if TASK_CONDITIONED:

        model = torch.load(f'../data/final_models/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}.pth', map_location=torch.device(DEVICE))
        task_embedder = torch.load(f'../data/final_models/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}_embedder.pth', map_location=torch.device(DEVICE))

        model.eval()
        model.to(DEVICE)
        model.device = DEVICE
        if modality== 'kinesthetic': #clean this up with a model.move_to_device
            model.encoder.device = DEVICE
        task_embedder.eval()
        task_embedder.to(DEVICE)
        task_embedder.device = DEVICE

        mode_to_type = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
        df = pd.read_csv('../data/all_data.csv').query(f'type=="{mode_to_type[modality]}"')

        embeds = generate_all_embeddings_taskconditioned(model, task_embedder, 
                                    df, DEVICE, data_dir='../data', pretrained_embeds_array=np.load(em_path))
        
        np.save(f'../data/embeds/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}.npy', embeds)
