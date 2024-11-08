import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

palette = {
    'contrastive': '#ff91af',
    'contrastive+autoencoder': '#e05780',
    'contrastive+VAE': '#f7cad0',
    'VAE': '#b6e2d3',
    'random': '#8f7073',
    'autoencoder': '#86a79c'
}
method2label = {
        'random': 'Random',
        'autoencoder': 'AE',
        'VAE': 'VAE',
        'contrastive': 'CLEA',
        'contrastive+autoencoder': 'CLEA+AE',
        'contrastive+VAE': 'CLEA+VAE',  
    }


plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

for ax, modality in zip([ax1, ax2, ax3], ['visual', 'auditory', 'kinetic']):
    results = {
        
        'random': [],
        'autoencoder': [],
        'VAE': [],
        'contrastive': [],
        'contrastive+autoencoder': [],
        'contrastive+VAE': [],
        
    }
    data = pd.read_csv(f'./linear_results.csv')
    data = data.query(f'modality == "{modality}" and dim_embedding == 8')

    print(data['pid'])


    for i, row in data.iterrows():
        m = np.fromstring(row['m'][1:-1], sep=' ')

        results[row['method']].append(m)

    for method, values in results.items():  
        color = palette[method]

        m = np.mean(values, axis=0)
        std = np.std(values, axis=0)

        N = 14
        ax.fill_between(range(101), m-(std/np.sqrt(N)), m+(std/np.sqrt(N)), alpha=0.2, color=color)
        ax.plot(m, label=method2label[method], color=color)
        if 'contrastive' in method:
            ax.plot(range(101)[::5], m[::5], color='#ffcc00', linestyle=' ', marker='o', markersize=2.5)
            

    ax.set_axisbelow(True)
    ax.grid(color='#DEDEDE', linestyle='dashed')
    ax.set_title(modality.capitalize())
    ax.set_xlabel('Number of Queries')
    ax.set_ylim(-.1, .65)
    ax.set_xlim(0, 100)
    
    
ax1.set_ylabel('Alignment')
ax2.set_yticklabels([])
ax3.set_yticklabels([])
# plt.ylabel('Alignment')
# plt.xlabel('Number of Queries')


plt.legend(ncol=6, bbox_to_anchor=(0.5, -.15))
plt.subplots_adjust(bottom=0.2, wspace=0.1, right=0.95, left=0.1)
sns.despine()
plt.tight_layout()
plt.show()