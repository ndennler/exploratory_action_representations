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
sns.countplot(data=pd.DataFrame(results), x='method')
plt.show()