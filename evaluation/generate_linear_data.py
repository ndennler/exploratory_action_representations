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
NUM_TRIALS = 20
dim_embedding = 64
TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}



results = []

for dim_embedding in [8,16,32,64,128]:
    #User Input and Estimation of reward functions
    user_choice_model = LuceShepardChoice()
    user_estimate = MonteCarloLinearReward(dim_embedding, number_samples=10_000)

    for method in ['contrastive', 'contrastive+autoencoder', 'contrastive+VAE', 'VAE', 'random', 'autoencoder']:
        for modality in ['auditory', 'visual', 'kinetic']:
            cumulative_values = []
            
            for participant_fname in os.listdir('../data_collection/processed_results'):
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

                for _ in tqdm(range(NUM_TRIALS)):
                    user_estimate.reset()
                    alignment = [0]
                    
                    data = np.random.permutation(data)

                    for i, trial in enumerate(data[:100]):
                        query = embeds[trial, TASK_INDEX_MAPPING[signal]]
                        user_choice_model.tell_input(1, query)
                        user_estimate.update(user_choice_model.get_probability_of_input)

                        alignment.append(alignment_metric(user_estimate.get_expectation(), true_preference))

                        # if i % 10 == 0:
                        #     probs = []
                        #     for test_trial in test_data[:50]:
                        #         query = embeds[test_trial, TASK_INDEX_MAPPING[signal]]
                        #         p = user_choice_model.get_choice_probabilities(query, np.array([user_estimate.get_expectation()]))
                        #         probs.append(np.log(p[1]))

                        #     alignment.append(np.sum(probs))

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
pd.DataFrame(results).to_csv('linear_results_32.csv', index=False)