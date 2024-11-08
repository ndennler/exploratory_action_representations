import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import pandas as pd

from irlpreference.input_models import LuceShepardChoice, WeakPreferenceChoice
from irlpreference.query_generation import InfoGainQueryGenerator, RandomQueryGenerator, VolumeRemovalQueryGenerator
from irlpreference.reward_parameterizations import MonteCarloLinearReward

def alignment_metric(true_w, guessed_w):
    return np.dot(guessed_w, true_w) / (np.linalg.norm(guessed_w) * np.linalg.norm(true_w))

#Experimental Constants
NUM_TRIALS = 60
TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}



results = []

for dim_embedding in [8,16,32,64,128]:
    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

    for method in ['contrastive', 'contrastive+autoencoder', 'contrastive+VAE', 'VAE', 'random', 'autoencoder']:
        for modality in ['auditory', 'visual', 'kinetic']:
            cumulative_values = []
            
            for participant_fname in tqdm(os.listdir('../data_collection/processed_results')):
                pid, this_modality, signal = participant_fname[:-4].split('&')

                if modality != this_modality:
                    continue

                data = np.load(f'../data_collection/processed_results/{participant_fname}')
                
                if modality == 'kinetic':
                    embeds = np.load(f'../data/embeds/kinesthetic&independent&raw&{method}&all_signals&{dim_embedding}.npy')
                elif modality == 'visual':
                    embeds = np.load(f'../data/embeds/visual&independent&raw&{method}&all_signals&{dim_embedding}.npy')
                else:
                    embeds = np.load(f'../data/embeds/auditory&independent&raw&{method}&all_signals&{dim_embedding}.npy')


                true_preference = embeds[data['top_id'], TASK_INDEX_MAPPING[signal]]

                test_data = data['test']
                data = (data['train'])

                for noise_level in [0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.4]:
                    for _ in range(NUM_TRIALS):
                        user_estimate.reset()
                        
                        data = np.random.permutation(data)

                        for i, trial in enumerate(data[:100]):
                            query = embeds[trial, TASK_INDEX_MAPPING[signal]]

                            #add noise to test robustness
                            query += noise_level * np.random.randn(*query.shape)

                            user_choice_model.tell_input(1, query)
                            user_estimate.update(user_choice_model.get_probability_of_input)

                        results.append({
                            'method': method,
                            'dim_embedding': dim_embedding,
                            'alignment': alignment_metric(user_estimate.get_expectation(), true_preference),
                            'modality': modality,
                            'noise_level': noise_level,
                            'signal': signal,
                            'pid': pid,
                        })

pd.DataFrame(results).to_csv('robustness_results.csv', index=False)