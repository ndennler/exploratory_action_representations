import pandas as pd
import torch
from clea.dataloaders.exploratory_loaders import RawChoiceDatasetwithTaskEmbedding
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from tqdm import tqdm
import numpy as np
import random

def get_dataloader(batch_size: int, modality: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}"')
    dataset = TESTDATASET(df, kind=modality, transform=torch.Tensor, data_dir='../data/')
    embedding_dataloader = DataLoader(dataset, batch_size=batch_size)

    return embedding_dataloader, dataset.get_input_dim()


TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

class TESTDATASET(Dataset):
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
            self.stimulus_array = self.preload_data()

        elif kind == 'auditory':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Audio"')
            self.stimulus_directory = data_dir + 'auditory/aud/'
            self.stimulus_array = self.preload_data()

        elif kind == 'kinesthetic':
            self.stimulus_mapping = pd.read_csv(self.indexer_csv_location).query('type=="Movement"')
            self.stimulus_array = np.load(data_dir + 'kinetic/behaviors.npy')
            
        

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

        for id in stim_ids:
            print(id)
            fname = stimulus_mapping.query(f'id=={id}')['file'].values[0]
            path = stimulus_directory + fname[:-4] + '.jpg'
            with Image.open(path) as im:
                im = np.array(im) / 255.0
                im = np.moveaxis(im, -1, 0)
                database[id] = im

        return database # dict of all the stimulus, preloaded and scaled correctly


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
            # anchor = np.array(Image.open(self.get_stimulus_fname(int(anchor_index)))) / 255.0
            # positive = np.array(Image.open(self.get_stimulus_fname(int(positive_index)))) / 255.0
            # negative = np.array(Image.open(self.get_stimulus_fname(int(negative_index)))) / 255.0
            
            # anchor, positive, negative = (np.moveaxis(data, -1, 0) for data in [anchor, positive, negative])
            anchor = self.stimulus_array[int(anchor_index)]
            positive  = self.stimulus_array[int(positive_index)]
            negative = self.stimulus_array[int(negative_index)] 

        elif self.kind == 'kinesthetic':
            anchor = self.stimulus_array[self.get_stimulus_fname(int(anchor_index)), :] * 25
            positive  = self.stimulus_array[self.get_stimulus_fname(int(positive_index)), :] * 25
            negative = self.stimulus_array[self.get_stimulus_fname(int(negative_index)), :] * 25

        # if self.pretrained_embed_path is None:
        return self.transform(anchor),self.transform(positive),self.transform(negative), torch.tensor(signal_index)
        
        # else:
        #     a_embed = self.pretrained_embed[int(anchor),:]
        #     p_embed = self.pretrained_embed[int(positive),:]
        #     n_embed = self.pretrained_embed[int(negative),:]

        #     return self.transform(a_embed),self.transform(p_embed),self.transform(n_embed), \
        #             self.transform(anchor),self.transform(positive),self.transform(negative), \
        #             torch.tensor(signal_index), torch.tensor(self.pretrained_embed[int(item),:])





dataloader, input = get_dataloader(32, 'auditory')

for _ in tqdm(range(100)):
    for x,y,z, w in dataloader:
        print(x.shape)
