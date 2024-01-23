import torch
import pandas as pd
from matplotlib import pyplot as plt
import torchvision
import numpy as np

from clea.dataloaders.exploratory_loaders import RawChoiceDatasetwithTaskEmbedding
from clea.representation_models.train_model_utils import MultiEpochsDataLoader
from clea.dataloaders.exploratory_loaders import TASK_INDEX_MAPPING

def get_dataloader(batch_size: int, modality: str):
    df = pd.read_csv('../data/plays_and_options.csv') #TODO: make this changeable
    df = df.query(f'type == "{modality}"')
    dataset = RawChoiceDatasetwithTaskEmbedding(df, train=True, kind=modality, transform=torch.Tensor, data_dir='../data/')
    embedding_dataloader = MultiEpochsDataLoader(dataset, batch_size=batch_size, num_workers=4, persistent_workers=True, shuffle=True)

    return embedding_dataloader, dataset.get_input_dim()

if __name__ == "__main__":
    N_EXAMPLES = 6
    device = 'mps'
    modality = 'auditory'

    data, loss = get_dataloader(N_EXAMPLES, modality)

    model_name = f'taskemb_{modality}_autoencoder_64'
    model = torch.load('../data/trained_models/' + model_name + '.pth')
    model.eval()
    model.to(device)

    if 'taskemb' in model_name:
        task_embedder = torch.load('../data/trained_models/' + model_name + '_embedder.pth')
        task_embedder.eval()
        task_embedder.to(device)

    
    for batch_idx, (anchor, positive, negative, task_idxs) in enumerate(data):
        print(batch_idx)

        a_embed = model.encode(anchor.to(device))
        p_embed = model.encode(positive.to(device))
        n_embed = model.encode(negative.to(device))

        a_embed = task_embedder(a_embed, task_idxs)
        p_embed = task_embedder(p_embed, task_idxs)
        n_embed = task_embedder(n_embed, task_idxs)

        a_recon = model.decode(a_embed)
        p_recon = model.decode(p_embed)
        n_recon = model.decode(n_embed)
        
        grid_img = torchvision.utils.make_grid(a_recon, nrow=N_EXAMPLES)
        im1 = grid_img.permute(1, 2, 0).cpu()

        grid_img = torchvision.utils.make_grid(anchor, nrow=N_EXAMPLES)
        im2 = grid_img.permute(1, 2, 0).cpu() 

        grid_img = torchvision.utils.make_grid(p_recon, nrow=N_EXAMPLES)
        im3 = grid_img.permute(1, 2, 0).cpu()

        grid_img = torchvision.utils.make_grid(positive, nrow=N_EXAMPLES)
        im4 = grid_img.permute(1, 2, 0).cpu()   

        grid_img = torchvision.utils.make_grid(n_recon, nrow=N_EXAMPLES)
        im5 = grid_img.permute(1, 2, 0).cpu()

        grid_img = torchvision.utils.make_grid(negative, nrow=N_EXAMPLES)
        im6 = grid_img.permute(1, 2, 0).cpu()

        print(im1.shape, im2.shape)
        print(TASK_INDEX_MAPPING[task] for task in task_idxs)
        plt.imshow(np.concatenate([im1, im2, im3, im4, im5, im6], axis=0)) #goes top to bottom: anchor, positive, negative
        plt.show()

        break