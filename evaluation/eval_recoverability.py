import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('nn_results.csv')
EM = 128

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
ax = sns.barplot(data=df.query(f'embedding_size == {EM}'), x='modality', y='accuracy', hue='method', 
                 hue_order=hue_order, palette=palette, capsize=.1, errwidth=0.8, 
                 order=['visual', 'auditory', 'kinetic'])

# Set grid and labels
ax.set_axisbelow(True)
ax.grid(color='#DEDEDE', linestyle='dashed')
plt.xlabel('')
plt.ylabel('Predicted Choice Accuracy')
plt.ylim(0.5, 1)

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
