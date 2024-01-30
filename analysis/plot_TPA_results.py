import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import pingouin as pg
import numpy as np

sns.set_theme(style="ticks", font_scale=1.3)
plt.figure(figsize=(5,6))

df = pd.read_csv('../data/results/taskembedding_TPA_results.csv')

# means = df.groupby(['modality', 'embedding_type', 'seed'])['metric'].mean()
# means = means.reset_index()

# translation_dict = {'contrastive+autoencoder': 'CLEA+AE (ours)', 'contrastive': 'CLEA (ours)', 'autoencoder': 'AE', 'random': 'Random', 'VAE': 'VAE'}
# means['embedding_type'] = means['embedding_type'].replace(translation_dict)

# translation_dict = {'kinesthetic': 'Kinetic', 'auditory': 'Auditory', 'visual': 'Visual'}
# means['modality'] = means['modality'].replace(translation_dict)

# print(means)

# print('\n---------- Visual Results -----------\n\n')
# print(pg.welch_anova(data=means[means['modality'] == 'Visual'], dv='metric', between='embedding_type'))

# print('\n---------- Auditory Results -----------\n\n')
# print(pg.welch_anova(data=means[means['modality'] == 'Auditory'], dv='metric', between='embedding_type'))

# print('\n---------- Kinetic Results -----------\n\n')
# print(pg.welch_anova(data=means[means['modality'] == 'Kinetic'], dv='metric', between='embedding_type'))


# print(df.groupby(['embedding_type', 'modality', 'signal'])['metric'].mean())

print(df)

ax = sns.barplot(data=df, x='embed_name', y='metric', hue='embedding_type')
# sns.lineplot(data=df, x='embedding_size', y='metric', hue='embedding_type', errorbar='se')
plt.ylabel('Choice Prediction Accuracy')
plt.xlabel('Modality')

legend = plt.legend(ncol=2)
legend.set_bbox_to_anchor((1.06, 1.3)) 
legend.set_title("")
plt.tight_layout()
plt.show()

# means = means.dropna()

# ms = means.groupby(['embedding_type', 'modality', 'signal'])['metric'].mean()
# ms = ms.reset_index()

# stds = means.groupby(['embedding_type', 'modality', 'signal'])['metric'].std()
# std = stds.reset_index()
# print(ms)

# for modality in ['Visual', 'Auditory', 'Kinetic']:
#     for method in ['Random', 'VAE', 'AE', 'CLEA (ours)', 'CLEA+AE (ours)']:
#         string = modality + ' & ' + method + ' & '
    
#         for signal in ['idle', 'searching', 'has_item', 'has_information']:
#             m = ms.query(f'embedding_type == "{method}" & modality == "{modality}" & signal == "{signal}"')['metric'].values[0]
#             s = std.query(f'embedding_type == "{method}" & modality == "{modality}" & signal == "{signal}"')['metric'].values[0]

#             string += '\small' + f'{m:.2f} \\tiny $\pm$ '.lstrip('0') +  f'{s:.2f} &'.lstrip('0')

#         print(string[:-2] + '\\\\' )
#         print("\n")


# print(means.groupby(['embedding_type', 'modality'])['metric'].mean())