import time
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

class QueryDataset(Dataset):
    '''
    Dataset to load all query data
    '''
    def __init__(self, df, train=True, transform=None, kind='visual'):
        self.is_train = train
        self.transform = transform
        self.kind = kind
        
        if kind == 'visual':
            self.embeddings = np.load('../data/visual_large_embeddings.npy')
            
        self.data = df
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        options = [int(index) for index in d['query'].split(',')]
        selected_index = int(d['choice'])

        return self.transform(self.embeddings[options[0]]),  \
                self.transform(self.embeddings[options[1]]), \
                self.transform(self.embeddings[options[2]]), \
                torch.tensor(selected_index)

        
if __name__ == '__main__':
    kind = 'visual'
    df = pd.read_csv('../data/train_queries.csv')
    df = df.query(f'type == "{kind}"')

    dataset = QueryDataset(df,train=True, kind='visual', transform=torch.Tensor)

    dl = DataLoader(dataset, batch_size=32)
    for x,y,z,w in dl:
        print(x.shape)
        break