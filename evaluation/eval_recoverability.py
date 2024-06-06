import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('nn_results.csv')
# df = df.query('modality != "auditory"')
EM = 128

fig = plt.figure(figsize=(6.5, 5.5))
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

name2label = {
    'random': 'Random',
    'autoencoder': 'AE',
    'VAE': 'VAE',
    'contrastive': 'CLEA',
    'contrastive+autoencoder': 'CLEA+AE',
    'contrastive+VAE': 'CLEA+VAE'
}

df['method'] = df['method'].apply(lambda x: name2label[x])

hue_order = ['random', 'autoencoder', 'VAE', 'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']
hue_order = [name2label[h] for h in hue_order]
palette = ['#999999', '#0033cc', '#3399ff', 'tab:orange', '#ff6600', '#ff9966']

palette = {
    'CLEA': 'tab:orange',
    'CLEA+AE': '#d62b0d',
    'CLEA+VAE': '#decb3a',
    'VAE': '#3399ff',
    'Random': '#999999',
    'AE': '#0033cc'
}
ax = sns.barplot(data=df.query(f'embedding_size == {EM}'), x='modality', y='accuracy', hue='method', 
            hue_order=hue_order, palette=palette, capsize=.1, errwidth=0.8, order=['visual', 'auditory','kinetic'])


ax.set_axisbelow(True)
ax.grid(color='#DEDEDE', linestyle='dashed')
plt.xlabel('')
plt.ylabel('Predicted Choice Accuracy')
plt.ylim(0.5, 1)

reorder=lambda hl,nc:(sum((lis[i::nc]for i in range(nc)),[])for lis in hl)
h_l = ax.get_legend_handles_labels()
ax.legend(*reorder(h_l, 3), ncol=3, bbox_to_anchor=(.5, 1.2), loc='upper center')

# plt.legend(ncol=6, bbox_to_anchor=(1.1, 1.15), prop={'size': 10})
plt.xticks([0, 1, 2], ['Visual', 'Auditory', 'Kinetic'])

plt.tight_layout()
plt.show()