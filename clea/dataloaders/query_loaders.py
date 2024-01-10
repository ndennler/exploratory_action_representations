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
from PIL import Image

 #TODO: make this easier to change
TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

class QueryDataset(Dataset):
    '''
    Dataset to load all query data
    '''
    def __init__(self, df, train=True, transform=None, kind='visual'):
        self.is_train = train
        self.transform = transform
        self.kind = kind
        
        if kind == 'visual':
            self.embeddings = np.load('./data/visual/visual_pretrained_embeddings.npy')
        if kind == 'auditory':
            self.embeddings = np.load('./data/auditory/auditory_pretrained_embeddings.npy')
        if kind == 'kinesthetic':
            self.embeddings = np.load('./data/kinetic/kinetic_pretrained_embeddings.npy')
            
        self.data = df

    def get_single_item_by_index(self, index):
        return self.transform(self.embeddings[index]).unsqueeze(0)
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        options = [int(index) for index in d['query'].split(',')]
        selected_index = int(d['choice'])

        return self.transform(self.embeddings[options[0]]),  \
                self.transform(self.embeddings[options[1]]), \
                self.transform(self.embeddings[options[2]]), \
                torch.tensor(selected_index), options
    
    


class RawQueryDataset(Dataset):
    def __init__(self, df, train=True, transform=None, kind='visual', data_dir='./data/', task_embedding=False):
        self.is_train = train
        self.transform = transform
        self.kind = kind

        self.task_embedding = task_embedding
        self.indexer_csv_location = data_dir + 'all_data.csv'

        
        if kind == 'visual':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Video"')
            self.stimulus_directory = data_dir + 'visual/vis/'


        elif kind == 'auditory':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Audio"')
            self.stimulus_directory = data_dir + 'auditory/aud/'

        elif kind == 'kinesthetic':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Movement"')
            self.stimulus_array = np.load(data_dir + 'kinetic/behaviors.npy')
            
        self.data = df
    
    def get_stimulus_fname(self, index):
        if self.kind == 'visual':
            name = self.stimulus_mapping.query(f'id=={index}').file.values[0]
            name = self.stimulus_directory + name.replace('mp4', 'jpg')

        elif self.kind == 'auditory':
            name = self.stimulus_mapping.query(f'id=={index}').file.values[0]
            name = self.stimulus_directory + name.replace('wav', 'jpg')

        elif self.kind == 'kinesthetic':
            name = int(index)
    
        return name

    def get_input_dim(self):
        if self.kind in ['auditory', 'visual']:
            im = Image.open(self.get_stimulus_fname(0))
            return np.moveaxis(np.array(im),-1,0).shape
        elif self.kind == 'kinesthetic':
            return self.stimulus_array[self.get_stimulus_fname(0), :].shape
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        options = [int(index) for index in d['query'].split(',')]
        selected_index = int(d['choice'])

        if self.kind in ['auditory', 'visual']:
                option1 = np.array(Image.open(self.get_stimulus_fname(options[0]))) / 255.0
                option2 = np.array(Image.open(self.get_stimulus_fname(options[1]))) / 255.0
                option3 = np.array(Image.open(self.get_stimulus_fname(options[2]))) / 255.0
                
                option1,option2,option3 = (np.moveaxis(data, -1, 0) for data in [option1,option2,option3])

        elif self.kind == 'kinesthetic':
                option1 = self.stimulus_array[self.get_stimulus_fname(options[0]), :]
                option2  = self.stimulus_array[self.get_stimulus_fname(options[1]), :]
                option3 = self.stimulus_array[self.get_stimulus_fname(options[2]), :]

        if self.task_embedding:
            self.task_to_index_mapping = TASK_INDEX_MAPPING
            return self.transform(option1),  \
                   self.transform(option2), \
                   self.transform(option3), \
                   torch.tensor(selected_index), options, \
                   torch.tensor(self.task_to_index_mapping[d['signal']])
        
        return self.transform(option1), \
               self.transform(option2), \
               self.transform(option3), \
               torch.tensor(selected_index), options
    


class CachedRawQueryDataset(Dataset):
    '''
    Dataset to load all query data
    '''
    def __init__(self, df, train=True, transform=None, name=None):
        self.is_train = train
        self.transform = transform
        
        self.embeddings = np.load(f'./data/embeds/{name}.npy')
            
        self.data = df

    def get_single_item_by_index(self, index):
        return self.transform(self.embeddings[index]).unsqueeze(0)
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        options = [int(index) for index in d['query'].split(',')]
        selected_index = int(d['choice'])

        return self.transform(self.embeddings[options[0]]),  \
                self.transform(self.embeddings[options[1]]), \
                self.transform(self.embeddings[options[2]]), \
                torch.tensor(selected_index), options
    


class CachedRawQueryTaskEmbeddingDataset(Dataset):
    '''
    Dataset to load all query data
    '''
    def __init__(self, df, train=True, transform=None, name=None):
        self.is_train = train
        self.transform = transform
        
        self.embeddings = np.load(f'./data/embeds/{name}.npy')
            
        self.data = df
        self.task_to_index_mapping = TASK_INDEX_MAPPING

    def get_single_item_by_index(self, index):
        return self.transform(self.embeddings[index]).unsqueeze(0)
      
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        options = [int(index) for index in d['query'].split(',')]
        selected_index = int(d['choice'])
        task_idx = int(self.task_to_index_mapping[d['signal']])

        #returns embeddings for specific task index
        return self.transform(self.embeddings[options[0], task_idx]),  \
                self.transform(self.embeddings[options[1], task_idx]), \
                self.transform(self.embeddings[options[2], task_idx]), \
                torch.tensor(selected_index), options
    

        
if __name__ == '__main__':
    kind = 'visual'
    df = pd.read_csv('../data/train_queries.csv')
    df = df.query(f'type == "{kind}"')

    dataset = RawQueryDataset(df,train=True, kind='visual', transform=torch.Tensor)

    dl = DataLoader(dataset, batch_size=32)
    for x,y,z,w in dl:
        print(x.shape)
        break

