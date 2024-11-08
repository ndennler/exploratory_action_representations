import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


palette = {
    'CLEA': '#ff91af',
    'CLEA+AE': '#e05780',
    'CLEA+VAE': '#f7cad0',
    'VAE': '#b6e2d3',
    'Random': '#8f7073',
    'AE': '#86a79c'
}
method2label = {
        'contrastive': 'CLEA',
        'contrastive+autoencoder': 'CLEA+AE',
        'contrastive+VAE': 'CLEA+VAE',
        'VAE': 'VAE',
        'random': 'Random',
        'autoencoder': 'AE'
    }

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))


data = pd.read_csv(f'./robustness_results.csv')
data['method'] = data['method'].apply(lambda x: method2label[x])
for ax, modality in zip([ax1, ax2, ax3], ['visual', 'auditory', 'kinetic']):
    df = data.query(f'modality == "{modality}" and dim_embedding == 8')
    sns.lineplot(data=df, x='noise_level', y='alignment', hue='method', palette=palette, ax=ax)

    ax.set_axisbelow(True)
    ax.grid(color='#DEDEDE', linestyle='dashed')
    ax.set_title(modality.capitalize())
    ax.set_xlabel('Noise Level')
    ax.set_ylabel('')
    ax.set_ylim(-.1, .6)
    ax.set_xlim(0, 0.4)
    ax.legend([],[], frameon=False)

ax1.set_ylabel('Alignment')
ax2.set_yticklabels([])
ax3.set_yticklabels([])

# reorder=lambda hl,nc:(sum((lis[i::nc]for i in range(nc)),[])for lis in hl)
# h_l = ax.get_legend_handles_labels()
# plt.legend(*reorder(h_l, 3), ncol=6, bbox_to_anchor=(.5, 1.2), loc='upper center')
handles, labels = plt.gca().get_legend_handles_labels()
order = [4,5,3,0,1,2]
plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],ncol=6, bbox_to_anchor=(0.5, -.15),)


plt.subplots_adjust(bottom=0.2, wspace=0.1, right=0.95, left=0.1)
sns.despine()
plt.tight_layout()
plt.show()