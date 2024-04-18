import pandas as pd
import numpy as np
# from collections import set
data = pd.read_csv('../data/plays_and_options.csv')

print(data)

visual_set = set()
auditory_set = set()
kinesthetic_set = set()

for index, row in data.iterrows():
    if row['type'] == 'visual':
        for id in row['options'].split(','):
            visual_set.add(int(id))
        for id in row['chosen'].split(','):
            visual_set.add(int(id))

    if row['type'] == 'auditory':
        for id in row['options'].split(','):
            auditory_set.add(int(id))
        for id in row['chosen'].split(','):
            auditory_set.add(int(id))

    if row['type'] == 'kinesthetic':
        for id in row['options'].split(','):
            if int(id) % 5 != 4:
                kinesthetic_set.add(int(id))
        for id in row['chosen'].split(','):
            if int(id) % 5 != 4:
                kinesthetic_set.add(int(id))
        

np.savez('./valid_ids.npz', visual=list(visual_set), auditory=list(auditory_set), kinesthetic=list(kinesthetic_set))