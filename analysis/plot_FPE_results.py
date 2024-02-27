import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

data = pd.read_csv('../data/results/FPE_results.csv')
data = data.query('pretraining != "raw" and conditioning == "taskconditioned" and embedding_size == 128')
data.sort_values(by=['modality'], inplace=True)
print(data.query('embedding_type == "random"').groupby(['modality'])['loss'].mean().values)

means = data.query('embedding_type == "random"').groupby(['modality'])['loss'].mean().values

# data.loc[data['modality'] == 'auditory', 'scaled_loss'] = data[data['modality'] == 'auditory']['loss'] / means[0]
# data.loc[data['modality'] == 'kinesthetic', 'scaled_loss'] = data[data['modality'] == 'kinesthetic']['loss'] / means[1]
# data.loc[data['modality'] == 'visual', 'scaled_loss'] = data[data['modality'] == 'visual']['loss'] / means[2]

print(data)
sns.barplot(x='pretraining', y='loss', hue='embedding_type', data=data)
plt.ylabel('Loss (Normalized)')
plt.xlabel('Modality')
legend = plt.legend(title='Embedding Type')
# legend.set_bbox_to_anchor((1.5, 1.06)) 
plt.show()


# print(data['loss'] / data.groupby('modality')['loss'].transform(max))