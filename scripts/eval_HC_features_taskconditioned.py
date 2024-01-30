import torch 
import torch.nn as nn
import torch.nn.functional as F
from clea.dataloaders import HCFeaturesTaskConditionedDataset
from clea.reward_models.model_definitions import HCFeatureLearner
from tqdm import tqdm
import pandas as pd

EMBEDDING_DIM = 64

results = []


experiments = [
    {'modality': 'visual', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'random', 'pretrained_embeds_path': '../data/visual/xclip_embeds.npy'},

    {'modality': 'visual', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},
    {'modality': 'visual', 'model_type': 'random', 'pretrained_embeds_path': '../data/visual/clip_embeds.npy'},

    # {'modality': 'auditory', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    # {'modality': 'auditory', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    # {'modality': 'auditory', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    # {'modality': 'auditory', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    # {'modality': 'auditory', 'model_type': 'random', 'pretrained_embeds_path': '../data/auditory/ast_embeds.npy'},
    
    # {'modality': 'auditory', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    # {'modality': 'auditory', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    # {'modality': 'auditory', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    # {'modality': 'auditory', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},
    # {'modality': 'auditory', 'model_type': 'random', 'pretrained_embeds_path': '../data/auditory/auditory_pretrained_embeddings.npy'},

    # {'modality': 'kinesthetic', 'model_type': 'contrastive', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    # {'modality': 'kinesthetic', 'model_type': 'autoencoder', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    # {'modality': 'kinesthetic', 'model_type': 'contrastive+autoencoder', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    # {'modality': 'kinesthetic', 'model_type': 'VAE', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
    # {'modality': 'kinesthetic', 'model_type': 'random', 'pretrained_embeds_path': '../data/kinetic/AE_embeds.npy'},
]



for _ in tqdm(range(5)):
    for e in experiments:
            modality, model_type, pretrained_embeds_path = e['modality'], e['model_type'], e['pretrained_embeds_path']
            embed_name = pretrained_embeds_path.split('/')[-1].split('.')[0]

            embeds_path = f'../data/embeds/taskconditioned_{embed_name}_{modality}_{model_type}_{EMBEDDING_DIM}.npy'
            dataset = HCFeaturesTaskConditionedDataset( embeds_path,
                                                        f'../data/handcrafted_features/{modality}.npy', transform=torch.Tensor)


            input_size = dataset[0][0].shape[0]
            output_size = dataset[0][1].shape[0]

            train, test = torch.utils.data.random_split(dataset, [.8, .2])

            train_data_loader = torch.utils.data.DataLoader(train, batch_size=32, shuffle=True)
            test_data_loader = torch.utils.data.DataLoader(test, batch_size=32, shuffle=True)


            feature_predictor = HCFeatureLearner(input_size, 1024, output_size, device='cpu')
            if modality == 'visual':
                # loss_fn = nn.BCEWithLogitsLoss(reduction='sum')
                loss_fn = nn.MSELoss(reduction='sum')
            elif modality == 'auditory' or modality == 'kinesthetic':
                loss_fn = nn.MSELoss(reduction='sum')

            optimizer = torch.optim.Adam(feature_predictor.parameters(), lr=1e-3)

            feature_predictor.train()

            for _ in tqdm(range(7)):
                for batch_idx, (embedding, hc_features) in enumerate(train_data_loader):
                    optimizer.zero_grad()

                    pred = feature_predictor(embedding)    

                    loss = loss_fn(pred, hc_features.float())
                    loss.backward()
                    optimizer.step()

            total_loss = 0
            for batch_idx, (embedding, hc_features) in enumerate(test_data_loader):
                pred = feature_predictor(embedding)
                loss = loss_fn(pred, hc_features)
                total_loss += loss.item()

            print(f'Average Loss : {total_loss / len(test_data_loader)}')

            results.append({
                'modality': modality,
                'embedding_type': model_type,
                'embedding_dim': EMBEDDING_DIM,
                'embed_name': embed_name,
                'loss': total_loss / len(test_data_loader)
            })

pd.DataFrame(results).to_csv('../data/results/HC_feature_predictor_results.csv')



    
