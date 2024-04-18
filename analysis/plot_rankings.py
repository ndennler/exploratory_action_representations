import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

dfs = []
for fname in os.listdir('../data_collection/results/'):
    dfs.append(pd.read_csv(f'../data_collection/results/{fname}'))

df = pd.concat(dfs)
df.sort_values('condition', inplace=True)
# df = df.query('signal == "idle"')

sns.barplot(x='modality', y='rank', hue='condition', data=df)

plt.show()