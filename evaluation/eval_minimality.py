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

    auc = np.trapezoid(m[:i_max+1], dx=1/100)

    plot_df.append({
        'method': row['method'],
        'embedding_size': int(row['dim_embedding']),
        'alignment': m[-1],
        'std': std[-1],
        'auc': auc,
        'modality': row['modality'],
    })


for modality in ['visual', 'auditory', 'kinetic']:
        
    df = pd.DataFrame(plot_df)
    df = df.query(f'modality == "{modality}"')

    palette = {
        'contrastive': 'tab:orange',
        'contrastive+autoencoder': '#d62b0d',
        'contrastive+VAE': '#decb3a',
        'VAE': '#3399ff',
        'random': '#999999',
        'autoencoder': '#0033cc'
    }



    ax = sns.lineplot(data=pd.DataFrame(df), x='embedding_size', y='auc', hue='method', err_style='bars', palette=palette, errorbar='se')
    plt.legend(bbox_to_anchor=(1.05, 1.15), ncol=3)
    plt.tight_layout()
    plt.show()

    print(modality)
    for method in ['random', 'autoencoder','VAE','contrastive', 'contrastive+autoencoder', 'contrastive+VAE']:
        # print(method)
        values = pd.DataFrame(df.query(f'method == "{method}"')).groupby(['embedding_size']).mean(numeric_only=True)['auc'].round(3).values
        values = [str(value).replace('-0.', '-.').replace('0.','.') for value in values]
        print(f'{method}:\t' + '\t'.join(map(str, values)))
        # print(pd.DataFrame(df.query(f'method == "{method}"')).groupby(['embedding_size']).sem().round(3))
    # print(pd.DataFrame(df).groupby(['method', 'embedding_size']).sem())