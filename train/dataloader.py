import torch
import random
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class ChoiceDataset(Dataset):
    def __init__(self, df, train=True, transform=None, kind='visual'):
        self.is_train = train
        self.transform = transform
        self.kind = kind
        
        if kind == 'visual':
            self.embeddings = np.load('../data/visual/visual_pretrained_embeddings.npy')
        elif kind == 'auditory':
            self.embeddings = np.load('../data/auditory/auditory_pretrained_embeddings.npy')
        elif kind == 'kinesthetic':
            self.embeddings = np.load('../data/kinetic/kinetic_pretrained_embeddings.npy')
            
        self.data = df
        
    def get_input_dim(self):
        return self.embeddings.shape[1]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        selected, unselected = d['chosen'].split(','), d['options'].split(',')

        # print(d)

        anchor_set = selected
        negative_set = unselected
        
        if len(selected) == 1: #swap the sets (exclusive or)
            anchor_set = unselected
            negative_set = selected
        elif np.random.rand() < 0.5 and len(unselected) > 1:
            anchor_set = unselected
            negative_set = selected

        anchor, positive = random.sample(anchor_set, 2)
        negative = random.sample(negative_set, 1)[0]
        
        if self.is_train:
            anchor, positive, negative = self.embeddings[int(anchor),:], self.embeddings[int(positive),:], self.embeddings[int(negative),:]

            return self.transform(anchor),self.transform(positive),self.transform(negative)


if __name__ == '__main__':
    kind = 'visual'
    df = pd.read_csv('../data/plays_and_options.csv')
    df = df.query(f'type == "{kind}"')

    dataset = ChoiceDataset(df,train=True, kind='visual', transform=torch.Tensor)

    dl = DataLoader(dataset, batch_size=32)
    for x,y,z in dl:
        print(x.shape)
        break