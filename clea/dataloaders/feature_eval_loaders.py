import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

class HCFeaturesDataset(Dataset):
    def __init__(self,  embeddings_path, handcrafted_features_path, transform=None):
        self.transform = transform
        self.embeddings = np.load(embeddings_path)
        self.handcrafted_features = np.load(handcrafted_features_path)
        

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, item):
        return self.transform(self.embeddings[item]), self.handcrafted_features[item]
    


class HCFeaturesTaskConditionedDataset(Dataset):
    def __init__(self,  embeddings_path, handcrafted_features_path, transform=None, task_idx=None):
        self.transform = transform
        self.embeddings = np.load(embeddings_path)
        self.handcrafted_features = np.load(handcrafted_features_path)
        self.task_idx = task_idx

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, item):
        if self.task_idx is None:
            task_idx = np.random.randint(0, 4)
        else:
            task_idx = self.task_idx
        return self.transform(self.embeddings[item, task_idx]), self.handcrafted_features[item]
    

if __name__ == "__main__":

    # test loading from the dataset with no task conditioning
    data = HCFeaturesDataset( '../../data/auditory/auditory_pretrained_embeddings.npy', 
                             '../../data/handcrafted_features/auditory.npy', transform=torch.Tensor)
    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    for batch_idx, (embedding, hc_features) in enumerate(data_loader):
        print(embedding.shape)
        print(hc_features.shape)
        break
    
    # test loading from the dataset with task conditioning
    data = HCFeaturesTaskConditionedDataset( '../../data/embeds/raw_taskembedding_auditory_random_128_task_embedder.pth.npy', 
                                            '../../data/handcrafted_features/auditory.npy', transform=torch.Tensor)
    data_loader = DataLoader(data, batch_size=32, shuffle=True)

    for batch_idx, (embedding, hc_features) in enumerate(data_loader):
        print(embedding.shape)
        print(hc_features.shape)
        break