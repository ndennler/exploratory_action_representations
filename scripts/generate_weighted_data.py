import pandas as pd
import numpy as np


data = pd.read_csv('../data/plays_and_options.csv', index_col=0)
data = data.drop_duplicates()

# weight the data
data['weight'] = data.groupby('pid').cumcount()
data['weight'] = data.groupby('pid')['weight'].transform(lambda x: np.linspace(0.1, 1, len(x)))

data.to_csv('../data/plays_and_options_weighted.csv')