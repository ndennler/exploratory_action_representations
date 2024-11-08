import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

from sklearn.random_projection import GaussianRandomProjection

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

#Experimental Constants
NUM_TRIALS = 20
dim_embedding = 64
TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}



results = []

for dim_embedding in [8,16,32,64,128]:
    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

    for method in ['pretrained']:
        for modality in ['auditory', 'visual', 'kinetic']:
            cumulative_values = []
            
            for participant_fname in os.listdir('../data_collection/processed_results'):
                pid, this_modality, signal = participant_fname[:-4].split('&')

                print(pid, this_modality, signal, modality)
                if modality != this_modality:
                    continue

                data = np.load(f'../data_collection/processed_results/{participant_fname}')
                
            
                if modality == 'kinetic':
                    embeds = np.load('../data/kinetic/xclip_embeds.npy')
                elif modality == 'visual':
                    embeds = np.load('../data/visual/xclip_embeds.npy')
                else:
                    embeds = np.load('../data/auditory/AST_embeds.npy')

                rp = GaussianRandomProjection(n_components=dim_embedding)
                reduced_array = rp.fit_transform(embeds)
                embeds = np.tile(reduced_array[:, np.newaxis, :], (1, 4, 1))


                true_preference = embeds[data['top_id'], TASK_INDEX_MAPPING[signal]]

                test_data = data['test']
                data = (data['train'])

                for _ in tqdm(range(NUM_TRIALS)):
                    user_estimate.reset()
                    alignment = [0]
                    
                    data = np.random.permutation(data)

                    for i, trial in enumerate(data[:100]):
                        query = embeds[trial, TASK_INDEX_MAPPING[signal]]
                        user_choice_model.tell_input(1, query)
                        user_estimate.update(user_choice_model.get_probability_of_input)

                        alignment.append(alignment_metric(user_estimate.get_expectation(), true_preference))

                    cumulative_values.append(alignment)

                m = np.mean(np.array(cumulative_values), axis=0) 
                std = np.std(np.array(cumulative_values), axis=0) 

                results.append({
                    'method': method,
                    'dim_embedding': dim_embedding,
                    'm': m,
                    'std': std,
                    'modality': modality,
                    'signal': signal,
                    'pid': pid,
                })

            # plt.fill_between(range(101), m-(std/np.sqrt(len(cumulative_values))), m+(std/np.sqrt(len(cumulative_values))), alpha=0.3)
            # plt.plot(m, label=method)

        # plt.legend()
        # plt.show()
df = pd.DataFrame(results)
df.to_csv('linear_pretrained_results.csv', index=False)

df.groupby(['modality', 'dim_embedding']).mean()

