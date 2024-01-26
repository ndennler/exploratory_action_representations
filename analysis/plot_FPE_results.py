import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/HC_feature_predictor_results.csv')
data.sort_values(by=['modality'], inplace=True)
print(data.groupby(['modality', 'embedding_type']).mean().query('embedding_type == "random"')['loss'].values)

means = data.groupby(['modality', 'embedding_type']).mean().query('embedding_type == "random"')['loss'].values

data.loc[data['modality'] == 'auditory', 'scaled_loss'] = data[data['modality'] == 'auditory']['loss'] / means[0]
data.loc[data['modality'] == 'kinesthetic', 'scaled_loss'] = data[data['modality'] == 'kinesthetic']['loss'] / means[1]
data.loc[data['modality'] == 'visual', 'scaled_loss'] = data[data['modality'] == 'visual']['loss'] / means[2]

print(data)
sns.barplot(x='modality', y='scaled_loss', hue='embedding_type', data=data)
plt.ylabel('Loss (Normalized)')
plt.xlabel('Modality')
legend = plt.legend(title='Embedding Type')
legend.set_bbox_to_anchor((1.5, 1.06)) 
plt.show()


# print(data['loss'] / data.groupby('modality')['loss'].transform(max))