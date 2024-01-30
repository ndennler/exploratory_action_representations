import torch
import pandas as pd
import numpy as np
from clea.reward_models.eval_utils import generate_all_embeddings_taskconditioned


MODALITY = 'auditory' #one of ['visual', 'auditory', 'kinesthetic']
TYPE = 'Audio' # one of ['Video', 'Audio', 'Movement']
DEVICE = 'cpu'
EMBEDDING_TYPE = 'random' 
EMBEDDING_SIZE = 64
TASK_CONDITIONED = True


for MODALITY, TYPE in [ ('kinesthetic', 'Movement')]:
    for EMBEDDING_TYPE in ['autoencoder', 'contrastive+autoencoder']:
        
        if TASK_CONDITIONED:
            model = torch.load(f'../data/trained_models/taskemb_{MODALITY}_{EMBEDDING_TYPE}_{EMBEDDING_SIZE}.pth', map_location=torch.device(DEVICE))
            task_embedder = torch.load(f'../data/trained_models/taskemb_{MODALITY}_{EMBEDDING_TYPE}_{EMBEDDING_SIZE}_embedder.pth', map_location=torch.device(DEVICE))

            model.eval()
            model.to(DEVICE)
            model.device = DEVICE
            if MODALITY == 'kinesthetic': #clean this up with a model.move_to_device
                model.encoder.device = DEVICE
            task_embedder.eval()
            task_embedder.to(DEVICE)
            task_embedder.device = DEVICE

            df = pd.read_csv('../data/all_data.csv').query(f'type=="{TYPE}"')

            embeds = generate_all_embeddings_taskconditioned(model, task_embedder, 
                                        df, DEVICE, data_dir='../data')
            
            np.save(f'../data/embeds/{MODALITY}_{EMBEDDING_TYPE}_{EMBEDDING_SIZE}_taskconditioned.npy', embeds)
