import pingouin as pg
import pandas as pd
import numpy as np


# First generate stats for recoverability

# df = pd.read_csv('results.csv')

# for modality in ['visual', 'auditory','kinetic']:
#     stats_df = df.query(f'modality == "{modality}" and embedding_size == 128')

#     print(f'\n\n{modality}\n\n')
    
#     print(pg.homoscedasticity(data=stats_df, dv='accuracy', group='method'))
#     print(stats_df.welch_anova(dv='accuracy', between='method').round(3))
#     print(pg.pairwise_gameshowell(data=stats_df, dv='accuracy',
#                         between='method').round(3))
    

# Minimality -- AUC?
df = pd.read_csv('linear_results.csv')
def get_auc(m):
    return np.fromstring(m[1:-1], sep=' ')[-1]

df['final_alignment'] = df['m'].apply(get_auc)

for modality in ['kinetic', 'visual', 'auditory']:
    stats_df = df.query(f'modality == "{modality}"')
    
    modified_df = []

    for method in stats_df['method'].unique():
        for i in range(14):
            curve = []
            for dim_em in stats_df['dim_embedding'].unique():
                mini_df = stats_df.query(f'dim_embedding == {dim_em} and method == "{method}"')
                curve.append(mini_df['final_alignment'].values[i])
            
            auc = np.trapz(curve)
            modified_df.append({
                'method': method,
                'AUC': auc
            })
    
    modified_df = pd.DataFrame(modified_df)

    print(f'\n\n{modality}\n\n')
    print(pg.homoscedasticity(data=modified_df, dv='AUC', group='method'))
    print(modified_df.welch_anova(dv='AUC', between='method').round(3))
    print(pg.pairwise_gameshowell(data=modified_df, dv='AUC',
                        between='method').round(3))


#Simplicity -- AUC
# def get_auc(m):
#     return np.trapz(np.fromstring(m[1:-1], sep=' '))

# df = pd.read_csv('linear_results.csv')
# df['AUC'] = df['m'].apply(get_auc)

# for modality in ['kinetic', 'visual', 'auditory']:
#     stats_df = df.query(f'modality == "{modality}" and dim_embedding == 8')

#     print(f'\n\n{modality}\n\n')
    
#     print(pg.homoscedasticity(data=stats_df, dv='AUC', group='method'))
#     print(stats_df.welch_anova(dv='AUC', between='method').round(3))
#     print(pg.pairwise_gameshowell(data=stats_df, dv='AUC',
#                         between='method').round(3))
