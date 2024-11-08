import pandas as pd

# Load the data
data = pd.read_csv('../data/plays_and_options.csv', index_col=0)
data.drop_duplicates(inplace=True)

print(data)

count_queries = 0
count_exploratory = 0

for i in range(len(data)):
    chosen = data.iloc[i]['chosen'].split(',')
    options = data.iloc[i]['options'].split(',')

    if len(chosen) + len(options) == 3:
        count_queries += 1
    else:
        count_exploratory += len(chosen)

print(f'Queries: {count_queries/(25)}')
print(f'Queries: {count_exploratory/(25)}')
