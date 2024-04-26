import pandas as pd
import numpy as np
import random

#change this between participants

PID = 24

condition_data = pd.read_csv('conditions.csv')
condition_data = condition_data.query('PID == @PID')

SIGNAL = condition_data['Signal'].values[0]
MODALITY = condition_data['Modality'].values[0]

print(SIGNAL, MODALITY)

NUM_TRIALS=10

valid_ids = np.load('./valid_ids.npz', allow_pickle=True)
print(valid_ids)

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}
MODALITY_EMBED_MAPPING = {'visual': 'clip_embeds', 'auditory': 'auditory_pretrained_embeddings', 'kinesthetic': 'AE_embeds'}
MODALITY_TYPE_MAPPING = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}

experiments = []

final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
all_data = pd.read_csv('../data/all_data.csv')

for modality in [MODALITY]:

    # get 10 random signals that people actually designed
    selected_indices = final_signals.query('signal == @SIGNAL')[modality].values
    selected_indices = np.random.choice(selected_indices[selected_indices >= 0], NUM_TRIALS, replace=False)
    
    
    for index in selected_indices:
        # each experiment consists of the next closest signal for each method
        experiment_indices = []
        methods = []
        paths = []

        for method in random.sample(['random', 'contrastive+autoencoder', 'VAE', 'autoencoder', 'contrastive', 'contrastive+VAE'], 5):
            embeds = np.load(f'../data/embeds/{modality}&taskconditioned&{MODALITY_EMBED_MAPPING[modality]}&{method}&all_signals&128.npy')
            
            target = embeds[index, TASK_INDEX_MAPPING[SIGNAL]]
            alignment = (target @ embeds[:, TASK_INDEX_MAPPING[SIGNAL]].T)
            norms = np.linalg.norm(embeds[:, TASK_INDEX_MAPPING[SIGNAL]], axis=1) * np.linalg.norm(target)
            alignment = alignment / norms

            sorted_ims = np.argsort(alignment)
            sorted_ims = sorted_ims[~np.isnan(alignment[sorted_ims])]


            for i in range(2, 20): #index -1 is the actual stimlulus itself
                if sorted_ims[-i] not in experiment_indices and sorted_ims[-i] in valid_ids[modality]:

                    print(sorted_ims[-i], sorted_ims[-i] in valid_ids[modality])

                    id = sorted_ims[-i]
                    experiment_indices.append(sorted_ims[-i])
                    methods.append(method)
                    if modality != 'kinesthetic':
                        paths.append(all_data.query(f'id == {id} and type == "{MODALITY_TYPE_MAPPING[modality]}"')['file'].values[0][:-4] + '.jpg')
                    else:
                        paths.append(f"{id}.png")
                    break
        
        experiments.append({
            'modality': modality if modality != 'kinesthetic' else 'kinetic',
            'signal': SIGNAL,
            'pid': PID,

            'id1':experiment_indices[0],
            'button1': paths[0],
            'condition1': methods[0],

            'id2':experiment_indices[1],
            'button2': paths[1],
            'condition2': methods[1],

            'id3':experiment_indices[2],
            'button3': paths[2],
            'condition3': methods[2],

            'id4':experiment_indices[3],
            'button4': paths[3],
            'condition4': methods[3],

            'id5':experiment_indices[4],
            'button5': paths[4],
            'condition5': methods[4],
        })


            
pd.DataFrame(experiments).sample(frac = 1).to_csv(f'experiment.csv', index=False)