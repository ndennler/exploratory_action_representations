import pandas as pd
import os
import numpy as np

for result in os.listdir('results'):
    if '.csv' not in result:
        continue

    df = pd.read_csv(f'./results/{result}')

    top_choice = df.query('rank == 4 and (trial == 9)').id.values[0]
    PID = df['pid'].values[0]
    signal = df['signal'].values[0]
    modality = df['modality'].values[0]
    print(PID,signal,modality)
    comps = []

    # get trial-specific comparisons
    for trial in range(10):
        trial_df = df.query('trial == @trial')

        #probably a better way to get all upper triangular indices but I am on a plane and have no wifi to look it up
        for pref in [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]:
            comps.append([trial_df['id'].values[pref[0]],trial_df['id'].values[pref[1]]])

    #add in the a < b and b<c -> a<c 's
    transitives = []
    for comp1 in comps:
        for comp2 in comps:
            if comp1[1] == comp2[0]:
                to_add = [comp1[0], comp2[1]]
                if to_add not in transitives:
                    transitives.append(to_add)

    dataset = comps #+ transitives
    train = int(len(dataset)*.7)
    np.random.shuffle(dataset)
    print(train, len(dataset) - train)
    np.savez(f'./processed_results/{PID}&{modality}&{signal}.npz', train=dataset[:train], test=dataset[train:], top_id=top_choice)
