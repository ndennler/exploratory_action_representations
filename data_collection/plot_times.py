import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

results = []

for f in os.listdir('results'):
    if '.csv' not in f:
        continue
    
    df = pd.read_csv(f'results/{f}')

    #gets average ranking of different methods (not including the super-rankings)
    # print(df.query('trial != 4 and trial != 9').groupby('condition')['rank'].mean())
    prev_time = None
    for i, row in df.query('rank == 4').iterrows():
        if prev_time is None:
            prev_time = row['time']
        else:
            results.append({'trial': row['trial'], 'time': (row['time'] - prev_time)/1000.})
            prev_time = row['time']


    #gets number of times each method was in the super rankings
    # print(df.query('trial == 4 or trial == 9').groupby('condition')['rank'].count())

    # number of times each method won the super rankings
    # print(df.query('(trial == 4 or trial == 9) and rank==4').groupby('condition')['rank'].count())
print(len(results))
df = pd.DataFrame(results)
sns.lineplot(data = df, x='trial', y='time')
plt.show()