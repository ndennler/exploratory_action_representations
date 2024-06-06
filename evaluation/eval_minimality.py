import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('linear_results.csv')

plot_df = []
for i, row in df.iterrows():
    m = np.fromstring(row['m'][1:-1], sep=' ')
    std = np.fromstring(row['std'][1:-1], sep=' ')
    i_max = np.argmax(m)

    plot_df.append({
        'method': row['method'],
        'embedding_size': int(row['dim_embedding']),
        'alignment': m[-1],
        'std': std[-1],
        'modality': row['modality'],
    })

plot_df = pd.DataFrame(plot_df)
plot_df = plot_df.query('modality == "visual"')

palette = {
    'contrastive': 'tab:orange',
    'contrastive+autoencoder': '#d62b0d',
    'contrastive+VAE': '#decb3a',
    'VAE': '#3399ff',
    'random': '#999999',
    'autoencoder': '#0033cc'
}



ax = sns.lineplot(data=pd.DataFrame(plot_df), x='embedding_size', y='alignment', hue='method', err_style='bars', palette=palette, errorbar='se')
plt.legend(bbox_to_anchor=(1.05, 1.15), ncol=3)
# plt.tight_layout()
# plt.show()

print(pd.DataFrame(plot_df).groupby(['method', 'embedding_size']).mean())
print(pd.DataFrame(plot_df).groupby(['method', 'embedding_size']).sem())