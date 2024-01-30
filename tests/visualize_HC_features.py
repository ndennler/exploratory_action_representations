import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


MODALITY = 'visual'


visual_concepts = {
    1 : ['baggage', 'hand', 'hands', 'luggage', 'box', 'package', 'parcel', 'bag'],
    2 : ['happy', 'smiling', 'smile', 'happiness',  'smiley'],
    3 : ['time', 'clock', 'wait', 'hour', 'day'],
    4 : ['sleep', 'relax', 'rest', 'eye'],
    5 : ['communications', 'message', 'chat', 'speech', 'bubble', 'conversation', 'talk'],
    6 : ['idea', 'lightbulb', 'information', 'info', 'brain', 'illumination'],
    7 : ['maps', 'map', 'location', 'arrow', 'pin', 'sign', 'transportation', 'vehicle', 'navigation', 'cursor', 'pointer'],
    8 : ['business', 'finance', 'card', 'payment', 'finance', 'banking'],
    9 : ['magnifying', 'glass', 'search', 'eye', 'radar', 'glasses'],
    10 : ['animal' , 'pet', 'animals'],
    11 : ['kid', 'baby', 'toy']
}

def calc_feature(filename, tags):
    total = 0

    for concept in visual_concepts[11]:
        if concept in tags:
            total += 1

    return total

def get_features(id, print_tags=False):
    filename, tags = all_data.query(f'id == {id}')[['file', 'tags']].values[0]
    tags = tags.split(',')
    if print_tags:
        print(tags)
    return calc_feature(filename, tags)

modality2type = {'auditory': 'Audio', 'visual': 'Video', 'kinesthetic': 'Movement'}

all_data = pd.read_csv('../data/all_data.csv')
final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')


for signal in ['idle']:
    plt.figure()

    selected_signals = final_signals.query(f'signal == "{signal}"')[MODALITY].values
    other_signals = np.random.choice(all_data.query(f'type == "{modality2type[MODALITY]}"')['id'].values, 80, replace=False)
    other_signals = [i for i in other_signals if i not in selected_signals]

    selected_features = [get_features(id, print_tags=True) for id in selected_signals]
    other_features = [get_features(id) for id in other_signals]

    plt.hist(selected_features, bins=20, alpha=0.5, label='Selected')
    plt.hist(other_features, bins=20, alpha=0.5, label='Other')
    plt.title(f'Signal: {signal}')
    plt.legend()
    plt.show()