import torch
import pandas as pd
import numpy as np
from clea.reward_models.eval_utils import generate_all_embeddings_taskconditioned, generate_all_embeddings_independent
import os 

EMBEDDING_DIM = 64
TASK_CONDITIONED = True
DEVICE = 'cpu'


def generate_taskconditioned_embeddings(model_path, em_path=None):
    modality, task_dependency, pretrained, embedding_type, signal, size = model_path[:-4].split('&')

    model = torch.load(f'../data/trained_models/{model_path}', map_location=torch.device(DEVICE))
    task_embedder = torch.load(f'../data/trained_models/{model_path[:-4]}_embedder.pth', map_location=torch.device(DEVICE))

    model.eval()
    model.to(DEVICE)
    model.device = DEVICE

    if modality== 'kinesthetic' and hasattr(model, 'encoder'): #clean this up with a model.move_to_device
        model.encoder.device = DEVICE
    task_embedder.eval()
    task_embedder.to(DEVICE)
    task_embedder.device = DEVICE

    mode_to_type = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
    df = pd.read_csv('../data/all_data.csv').query(f'type=="{mode_to_type[modality]}"')

    if em_path is not None:
        embeds = generate_all_embeddings_taskconditioned(model, task_embedder, 
                                df, DEVICE, data_dir='../data', pretrained_embeds_array=np.load(em_path))
    else:
        embeds = generate_all_embeddings_taskconditioned(model, task_embedder, 
                                    df, DEVICE, data_dir='../data')
    
    np.save(f'../data/embeds/{model_path[:-4]}.npy', embeds)




def generate_independent_embeddings(model_path, em_path=None):

    modality, task_dependency, pretrained, embedding_type, signal, size = model_path[:-4].split('&')
    model = torch.load(f'../data/trained_models/{model_path}', map_location=torch.device(DEVICE))

    model.eval()
    model.device = DEVICE
    if modality== 'kinesthetic' and hasattr(model, 'encoder'): #clean this up with a model.move_to_device
        model.encoder.device = DEVICE
    model.to(DEVICE)

    mode_to_type = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
    df = pd.read_csv('../data/all_data.csv').query(f'type=="{mode_to_type[modality]}"')

    if os.path.exists(f'../data/embeds/{modality}&{task_dependency}&{pretrained}&{embedding_type}&all_signals&{size}.npy'):
        embed_storage_path = f'../data/embeds/{modality}&{task_dependency}&{pretrained}&{embedding_type}&all_signals&{size}.npy'
    else:
        embed_storage_path = None

    # embeds = generate_all_raw_embeddings(model, df, DEVICE, EMBEDDING_DIM)
    if em_path is not None:
        embeds = generate_all_embeddings_independent(model=model, dataframe=df, embedding_size=int(size), signal=signal, 
                                                     device=DEVICE, data_dir='../data', pretrained_embeds_array=np.load(em_path), embed_storage_path=embed_storage_path)
    else:
        embeds = generate_all_embeddings_independent(model=model, dataframe=df, embedding_size=int(size), signal=signal, 
                                                     device=DEVICE, data_dir='../data', pretrained_embeds_array=None, embed_storage_path=embed_storage_path)


    np.save(f'../data/embeds/{modality}&{task_dependency}&{pretrained}&{embedding_type}&all_signals&{size}.npy', embeds)



for model_path in os.listdir('../data/trained_models'):
    if '.pth' not in model_path or 'embedder' in model_path:
        continue

    modality, task_dependency, pretrained, embedding_type, signal, size = model_path[:-4].split('&')

    #get the array of the pretrained embeddings as inputs if they were used
    if pretrained == 'raw':
            em_path = None
    else:
        em_path = f'../data/{modality}/{pretrained}.npy' if modality != 'kinesthetic' else f'../data/kinetic/{pretrained}.npy'

    if task_dependency == 'taskconditioned':
        generate_taskconditioned_embeddings(model_path, em_path=em_path)
    
    if task_dependency == 'independent':
        generate_independent_embeddings(model_path, em_path=em_path)
    

    





