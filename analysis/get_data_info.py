import pandas as pd
import numpy as np

# get data info for representation learning
df = pd.read_csv('../data/plays_and_options.csv')

print(df.columns)

print(f'Total number of queries made: {len(df["chosen"].unique())}')
print(f'Total number of participants: {len(df["pid"].unique())}')

print(f'Average number of played options: {np.mean([ len(q.split(",")) for q in df["chosen"].unique()])}')
print(f'Average number of presented options: {np.mean([ len(q.split(",")) for q in df["options"].unique()])}')

# get data info for reward learning (for evaluating choice accuracy)
df = pd.read_csv('../data/evaluation/all_queries.csv')
print(f'Total number of queries collected: {len(df)}')