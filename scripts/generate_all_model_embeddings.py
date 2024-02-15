import torch
import pandas as pd
import numpy as np
from clea.reward_models.eval_utils import generate_all_embeddings_taskconditioned, generate_all_raw_embeddings
import os 

EMBEDDING_DIM = 64
TASK_CONDITIONED = True
DEVICE = 'cpu'


def generate_taskconditioned_embeddings(modality, embed_name, model_type, em_path):
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


def generate_raw_embeddings(modality, signal, model_type):

    model_name = f'raw_{modality}_{model_type}_{signal}_{EMBEDDING_DIM}.pth'
    model = torch.load('../data/final_models/' + model_name, map_location=torch.device(DEVICE))

    model.eval()
    model.device = DEVICE
    if modality== 'kinesthetic' and hasattr(model, 'encoder'): #clean this up with a model.move_to_device
        model.encoder.device = DEVICE
    model.to(DEVICE)

    mode_to_type = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
    df = pd.read_csv('../data/all_data.csv').query(f'type=="{mode_to_type[modality]}"')

    embeds = generate_all_raw_embeddings(model, df, DEVICE, EMBEDDING_DIM)
    print(embeds)

    np.save(f'../data/embeds/raw_{modality}_{model_type}_{signal}_{EMBEDDING_DIM}.npy', embeds)



for model_path in os.listdir('../data/final_models'):
    if '.pth' not in model_path or 'embedder' in model_path:
        continue

    if 'taskconditioned' in model_path and 'embeds' in model_path:
        path_parts = model_path.split('_')
        embed_name = path_parts[1] + '_' + path_parts[2]
        modality = path_parts[3]
        model_type = path_parts[4]
        em_path = f'../data/{modality if modality != "kinesthetic" else "kinetic"}/{embed_name}.npy'

        if not os.path.exists(f'../data/embeds/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}.npy'):
            generate_taskconditioned_embeddings(modality, embed_name, model_type, em_path)
    
    if 'raw' in model_path:
        path_parts = model_path.split('_')
        modality = path_parts[1]
        model_type = path_parts[2]
        signal = path_parts[3] if len(path_parts) == 5 else path_parts[3] + '_' + path_parts[4]

        if not os.path.exists(f'../data/embeds/raw_{modality}_{model_type}_{signal}_{EMBEDDING_DIM}.npy'):
            generate_raw_embeddings(modality, signal, model_type)





