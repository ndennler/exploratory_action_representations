import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

results = []

for f in os.listdir('../data_collection/results'):
    if '.csv' not in f:
        continue
    
    df = pd.read_csv(f'../data_collection/results/{f}')

    #gets average ranking of different methods (not including the super-rankings)
    # print(df.query('trial != 4 and trial != 9').groupby('condition')['rank'].mean())
    for i, row in df.query('rank == 4 and (trial == 9)').iterrows():
        results.append({'method': row['condition'], 'modality': row['modality']})


    #gets number of times each method was in the super rankings
    # print(df.query('trial == 4 or trial == 9').groupby('condition')['rank'].count())

    # number of times each method won the super rankings
    # print(df.query('(trial == 4 or trial == 9) and rank==4').groupby('condition')['rank'].count())
print(len(results))

hue_order = ['random', 'autoencoder', 'VAE', 'contrastive', 'contrastive+autoencoder', 'contrastive+VAE']
palette = ['#999999', '#0033cc', '#3399ff', 'tab:orange', '#ff6600', '#ff9966']

palette = {
    'contrastive': 'tab:orange',
    'contrastive+autoencoder': '#d62b0d',
    'contrastive+VAE': '#decb3a',
    'VAE': '#3399ff',
    'random': '#999999',
    'autoencoder': '#0033cc'
}

plt.rcParams["font.family"] = "serif"
plt.rcParams['font.size'] = 14

ax= sns.countplot(data=pd.DataFrame(results), x='method', order=hue_order, palette=palette)

ax.set_axisbelow(True)
ax.grid(color='#DEDEDE', linestyle='dashed')

plt.axhline(y = 0.167 * len(results), color = '#999999', linestyle = '--') 
plt.xlabel('')
plt.ylabel('Number of Users')
ax = plt.gca()
ax.set_xticklabels(['Random', 'AE', 'VAE', 'CLEA', 'CLEA\n+AE', 'CLEA\n+VAE'])
plt.tight_layout()
plt.show()