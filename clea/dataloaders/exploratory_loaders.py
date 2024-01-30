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
from tqdm import tqdm

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


def preload_data(stimulus_directory, stimulus_mapping, exploratory_action_data):

        print('loading data...')

        stim_ids = set()
        ids = stimulus_mapping['id'].values

        for i, row in exploratory_action_data.iterrows():
            for id in row['chosen'].split(','):
                if int(id) in ids:
                    stim_ids.add(int(id))
            for id in row['options'].split(','):
                if int(id) in ids:
                    stim_ids.add(int(id))
        
        database = {}

        for id in tqdm(stim_ids):
            fname = stimulus_mapping.query(f'id=={id}')['file'].values[0]
            path = stimulus_directory + fname[:-4] + '.jpg'
            with Image.open(path) as im:
                im = np.array(im) / 255.0
                im = np.moveaxis(im, -1, 0)
                database[id] = im

        return database # dict of all the stimulus, preloaded and scaled correctly

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


class RawChoiceDataset(Dataset):
    def __init__(self, df, transform=None, kind='visual', data_dir='./data/'):

        self.data = df

        self.transform = transform
        self.kind = kind
        self.indexer_csv_location = data_dir + 'all_data.csv'
        
        if kind == 'visual':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Video"')
            self.stimulus_directory = data_dir + 'visual/vis/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'auditory':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Audio"')
            self.stimulus_directory = data_dir + 'auditory/aud/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'kinesthetic':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Movement"')
            self.stimulus_array = np.load(data_dir + 'kinetic/behaviors.npy')
    
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
        selected, unselected = d['chosen'].split(','), d['options'].split(',')

        anchor_set = selected
        negative_set = unselected
        
        if len(selected) == 1: #swap the sets (exclusive or)
            anchor_set = unselected
            negative_set = selected
        elif np.random.rand() < 0.5 and len(unselected) > 1:
            anchor_set = unselected
            negative_set = selected

        anchor_id, positive_id = random.sample(anchor_set, 2)
        negative_id = random.sample(negative_set, 1)[0]
    
        if self.kind in ['auditory', 'visual']:
            anchor = self.stimulus_array[int(anchor_id)]
            positive  = self.stimulus_array[int(positive_id)]
            negative = self.stimulus_array[int(negative_id)] 

        elif self.kind == 'kinesthetic':
            anchor = self.stimulus_array[self.get_stimulus_fname(int(anchor_id)), :] * 25
            positive  = self.stimulus_array[self.get_stimulus_fname(int(positive_id)), :] * 25
            negative = self.stimulus_array[self.get_stimulus_fname(int(negative_id)), :] * 25

        return self.transform(anchor),self.transform(positive),self.transform(negative)
        


class RawChoiceDatasetwithTaskEmbedding(Dataset):
    def __init__(self, df, transform=None, 
                 kind='visual', data_dir='./data/', 
                 pretrained_embed_path = None):
        
        self.data = df
        self.task_to_index_mapping = TASK_INDEX_MAPPING
        self.pretrained_embed_path = pretrained_embed_path

        self.transform = transform
        self.kind = kind
        self.indexer_csv_location = data_dir + 'all_data.csv'

        if pretrained_embed_path is not None:
            self.pretrained_embed = np.load(pretrained_embed_path)
        
        if kind == 'visual':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Video"')
            self.stimulus_directory = data_dir + 'visual/vis/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'auditory':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Audio"')
            self.stimulus_directory = data_dir + 'auditory/aud/'
            self.stimulus_array = preload_data(self.stimulus_directory, self.stimulus_mapping, self.data)

        elif kind == 'kinesthetic':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Movement"')
            self.stimulus_array = np.load(data_dir + 'kinetic/behaviors.npy')


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
        if self.pretrained_embed_path is not None:
            return self.pretrained_embed.shape[1]

        if self.kind in ['auditory', 'visual']:
            im = Image.open(self.get_stimulus_fname(0))
            return np.moveaxis(np.array(im),-1,0).shape
        elif self.kind == 'kinesthetic':
            return self.stimulus_array[self.get_stimulus_fname(0), :].shape
    
    def get_output_dim(self):
        if self.kind in ['auditory', 'visual']:
            im = Image.open(self.get_stimulus_fname(0))
            return np.moveaxis(np.array(im),-1,0).shape
        elif self.kind == 'kinesthetic':
            return self.stimulus_array[self.get_stimulus_fname(0), :].shape
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):

        d = self.data.iloc[item]
        selected, unselected, signal = d['chosen'].split(','), d['options'].split(','), d['signal']

        anchor_set = selected
        negative_set = unselected
        
        if len(selected) == 1: #swap the sets (exclusive or)
            anchor_set = unselected
            negative_set = selected
        elif np.random.rand() < 0.5 and len(unselected) > 1:
            anchor_set = unselected
            negative_set = selected

        anchor_index, positive_index = random.sample(anchor_set, 2)
        negative_index = random.sample(negative_set, 1)[0]
        
        signal_index = self.task_to_index_mapping[signal]

        if self.kind in ['auditory', 'visual']:
            anchor = self.stimulus_array[int(anchor_index)]
            positive  = self.stimulus_array[int(positive_index)]
            negative = self.stimulus_array[int(negative_index)] 

        elif self.kind == 'kinesthetic':
            anchor = self.stimulus_array[self.get_stimulus_fname(int(anchor_index)), :] * 25
            positive  = self.stimulus_array[self.get_stimulus_fname(int(positive_index)), :] * 25
            negative = self.stimulus_array[self.get_stimulus_fname(int(negative_index)), :] * 25

        if self.pretrained_embed_path is None:
            return self.transform(anchor),self.transform(positive),self.transform(negative), torch.tensor(signal_index)
        
        else:
            a_embed = self.pretrained_embed[int(anchor_index),:]
            p_embed = self.pretrained_embed[int(positive_index),:]
            n_embed = self.pretrained_embed[int(negative_index),:]

            return self.transform(a_embed),self.transform(p_embed),self.transform(n_embed), \
                    self.transform(anchor),self.transform(positive),self.transform(negative), \
                    torch.tensor(signal_index)

if __name__ == '__main__':
    kind = 'auditory'
    df = pd.read_csv('../../data/plays_and_options.csv')
    df = df.query(f'type == "{kind}"')

    dataset = RawChoiceDataset(df, kind=kind, transform=torch.Tensor, data_dir='../../data/')
    # print(len(dataset.preload_data()))
    print(dataset.get_stimulus_fname(0))
    print(dataset.get_input_dim())

    dl = DataLoader(dataset, batch_size=32)
    for x,y,z in dl:
        print(x.shape)
        break