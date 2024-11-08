import pandas as pd
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}
# load in ranking data from study 2
results = []

for f in os.listdir('../data_collection/results'):
    if '.csv' not in f:
        continue
    df = pd.read_csv(f'../data_collection/results/{f}')
    for i, row in df.query('rank == 4 and (trial == 9)').iterrows():
        if row['modality'] == 'kinetic':
            row['modality'] = 'kinesthetic'
        results.append({'id': row['id'], 'modality': row['modality'], 'signal': row['signal']})

print(len(results))

#get exemplar designed data from study 1
exemplar_data = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')



# get the closest exemplar for each method
min_distances = []
for method in ['random', 'contrastive+autoencoder', 'VAE', 'autoencoder', 'contrastive', 'contrastive+VAE']:
    for modality in ['visual', 'auditory', 'kinesthetic']:
        for result in results:
            if result['modality'] == modality:
                #get relevant exemplars
                exemplars = exemplar_data.query('signal == @result["signal"]')[modality].values
                exemplars = exemplars[exemplars >= 0]

                for em_size in [8, 16, 32, 64, 128]:
                    #get the embedding space
                    embeds = np.load(f'../data/embeds/{modality}&independent&raw&{method}&all_signals&64.npy')
                    
                    query_vector = embeds[result['id'], TASK_INDEX_MAPPING[result['signal']]]
                    exemplar_vectors = embeds[exemplars, TASK_INDEX_MAPPING[result['signal']]]

                    #calculate euclidean distances
                    distances = np.linalg.norm(exemplar_vectors - query_vector, axis=1) 
                    #calculate cosine similarities
                    distances = np.dot(exemplar_vectors, query_vector) / (np.linalg.norm(exemplar_vectors) * np.linalg.norm(query_vector))  
                    
                    mod= 'kinetic' if modality == 'kinesthetic' else modality

                    best = np.min(distances)
                    worst = np.max(distances)
                    min_distances.append({'dist' : worst , 'method': method, 'modality': mod})

min_distances = pd.DataFrame(min_distances)
df = min_distances

# sns.barplot(data=min_distances, x='modality', y='dist', hue='method', )
# plt.show()



fig = plt.figure(figsize=(6.5, 5.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

# Mapping method names to labels
name2label = {
    'random': 'Random',
    'autoencoder': 'AE',
    'VAE': 'VAE',
    'contrastive': 'CLEA',
    'contrastive+autoencoder': 'CLEA+AE',
    'contrastive+VAE': 'CLEA+VAE'
}

df['method'] = df['method'].apply(lambda x: name2label[x])

# Define the hue order and palette
hue_order = ['random', 'autoencoder', 'VAE', 'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']
hue_order = [name2label[h] for h in hue_order]
palette = {
    'CLEA': '#ff91af',
    'CLEA+AE': '#e05780',
    'CLEA+VAE': '#f7cad0',
    'VAE': '#b6e2d3',
    'Random': '#8f7073',
    'AE': '#86a79c'
}

# Create the bar plot
ax = sns.barplot(data=df, x='modality', y='dist', hue='method', 
                 hue_order=hue_order, palette=palette, capsize=.1, errwidth=0.8, errorbar='se',
                 order=['visual', 'auditory', 'kinetic'])

# Set grid and labels
ax.set_axisbelow(True)
ax.grid(color='#DEDEDE', linestyle='dashed')
plt.xlabel('')
plt.ylabel('Similarity to Closest Exemplar')
plt.ylim(0., 0.35)

# Add yellow outline around bars for methods containing "CLEA"
for i, patch in enumerate(ax.patches):
    print(patch.get_facecolor())

    if 'CLEA' in patch.get_label():
        patch.set_edgecolor('#ffcc00')  # Set edge color to yellow
        patch.set_linewidth(1.5)  # Set the edge width

    elif patch.get_facecolor()[0] > .8:
        patch.set_edgecolor('#ffcc00')  # Set edge color to yellow
        patch.set_linewidth(1.5)  # Set the edge width
    else:
        patch.set_edgecolor('none')  # No edge color for non-CLEA methods

# Reorder legend
reorder = lambda hl, nc: (sum((lis[i::nc] for i in range(nc)), []) for lis in hl)
h_l = ax.get_legend_handles_labels()
ax.legend(*reorder(h_l, 3), ncol=3, bbox_to_anchor=(.5, 1.2), loc='upper center')

sns.despine()
plt.tight_layout()
plt.show()

print(df.groupby(['method']).mean())