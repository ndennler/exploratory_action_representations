import pandas as pd
from PIL import Image
import numpy as np

MODALITY = 'Video'

all_data = pd.read_csv('../data/all_data.csv').query(f'type=="{MODALITY}"')
plays = pd.read_csv('../data/plays_and_options.csv')

stim_ids = set()
ids = all_data['id'].values

for i, row in plays.iterrows():
    for id in row['chosen'].split(','):
        if int(id) in ids:
            stim_ids.add(int(id))
    for id in row['options'].split(','):
        if int(id) in ids:
            stim_ids.add(int(id))

database = np.zeros((max(stim_ids)+1, 3, 224, 224))

for id in stim_ids:

    print(id)
    path = '../data/visual/vis/' + all_data.query(f'id=={id}')['file'].values[0].replace('.mp4', '.jpg')
    with Image.open(path) as im:
        im = np.array(im) / 255.0
        im = np.moveaxis(im, -1, 0)
        database[id] = im
    
    # print(database[id])

print(len(stim_ids))
print(len(all_data))


