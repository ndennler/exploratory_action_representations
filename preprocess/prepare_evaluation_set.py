import pandas as pd
import numpy as np
data = []

for pid in [2, 3, 4, 5, 6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]:
    choices = pd.read_csv(f'../../../personalization/data/{pid}/choices.csv')

    for i, row in choices.iterrows():
        data.append({
            'signal': row['signal'],
            'type': row['type'], 
            'query': row['query'],
            'choice': row['choice'],
            'pid': pid
        })
                
                


df = pd.DataFrame(data)
df['rand'] = np.random.rand(len(df))
df = df.sort_values('rand')
test_split = int(len(df) *.3)

train_df = df.iloc[test_split:, :5]
train_df = train_df.dropna()

test_df = df.iloc[:test_split, :5]
test_df = test_df.dropna()

print(len(test_df['pid'].unique()), len(train_df['pid'].unique()))

print(len(train_df), len(test_df))
train_df.to_csv('../data/train_queries.csv')
test_df.to_csv('../data/test_queries.csv')




