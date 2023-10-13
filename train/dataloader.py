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

class ChoiceDataset(Dataset):
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
        selected, unselected = d['chosen'].split(','), d['options'].split(',')

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

        # anchor_img = pass
        
        # if self.is_train:
        #     anchor_label = self.labels[item]

        #     positive_list = self.index[self.index!=item][self.labels[self.index!=item]==anchor_label]

        #     positive_item = random.choice(positive_list)
        #     positive_img = self.images[positive_item].reshape(28, 28, 1)
            
        #     negative_list = self.index[self.index!=item][self.labels[self.index!=item]!=anchor_label]
        #     negative_item = random.choice(negative_list)
        #     negative_img = self.images[negative_item].reshape(28, 28, 1)
            
        #     if self.transform:
        #         anchor_img = self.transform(self.to_pil(anchor_img))
        #         positive_img = self.transform(self.to_pil(positive_img))
        #         negative_img = self.transform(self.to_pil(negative_img))
            
        #     return anchor_img, positive_img, negative_img, anchor_label
        
        # else:
        #     if self.transform:
        #         anchor_img = self.transform(self.to_pil(anchor_img))
        #     return anchor_img
        
if __name__ == '__main__':
    kind = 'visual'
    df = pd.read_csv('../data/plays_and_options.csv')
    df = df.query(f'type == "{kind}"')

    dataset = ChoiceDataset(df,train=True, kind='visual', transform=torch.Tensor)

    dl = DataLoader(dataset, batch_size=32)
    for x,y,z in dl:
        print(x.shape)
        break