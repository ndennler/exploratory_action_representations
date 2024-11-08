import pingouin as pg
import pandas as pd
import numpy as np


# First generate stats for recoverability

# df = pd.read_csv('nn_results.csv')

# for modality in ['visual', 'auditory','kinetic']:
#     stats_df = df.query(f'modality == "{modality}" and embedding_size == 128')

#     print(f'\n\n{modality}\n\n')
    
#     # print(pg.homoscedasticity(data=stats_df, dv='accuracy', group='method'))
#     # print(stats_df.welch_anova(dv='accuracy', between='method').round(3))
#     # print(pg.pairwise_gameshowell(data=stats_df, dv='accuracy',
#     #                     between='method').round(3))

#     print(pg.normality(data=stats_df, dv='accuracy', group='method', method='jarque_bera').round(3))
#     print(pg.rm_anova(data=stats_df, dv='accuracy', within='method', subject='pid', correction=True, effsize='n2').round(3))
#     print(pg.pairwise_tests(data=stats_df, dv='accuracy', within='method', subject='pid', padjust='bonf', effsize='eta-square').round(3))
    

# Minimality -- AUC?
df = pd.read_csv('linear_results.csv')
def get_auc(m):
    return np.trapz(np.fromstring(m[1:-1], sep=' '), dx=1/100)

df['auc'] = df['m'].apply(get_auc)

print(df['auc'].values)


for modality in ['visual', 'auditory', 'kinetic']:
    stats_df = df.query(f'modality == "{modality}"')
    print(f'\n{modality}\n')
    print(pg.rm_anova(data=stats_df, dv='auc', within=['dim_embedding', 'method'], subject='pid', correction=True, effsize='n2').round(3))
    print(pg.pairwise_tests(data=stats_df, dv='auc', within='method', subject='pid', padjust='bonf', effsize='eta-square').round(3))

    
#     modified_df = []

#     for method in stats_df['method'].unique():
#         for i in range(14):
#             curve = []
#             for dim_em in stats_df['dim_embedding'].unique():
#                 mini_df = stats_df.query(f'dim_embedding == {dim_em} and method == "{method}"')
#                 curve.append(mini_df['final_alignment'].values[i])

#             auc = np.trapz(curve)
#             modified_df.append({
#                 'method': method,
#                 "dim": dim_em,
#                 'AUC': auc,
#                 'pid': i,
#             })
    
#     modified_df = pd.DataFrame(modified_df)

#     print(f'\n\n{modality}\n\n')
#     # print(pg.homoscedasticity(data=modified_df, dv='AUC', group='method'))
#     # print(modified_df.welch_anova(dv='AUC', between='method').round(3))
#     # print(pg.pairwise_gameshowell(data=modified_df, dv='AUC',
#     #                     between='method').round(3))
    
#     # print(pg.rm_anova(data=modified_df, dv='AUC', within=['dim', 'method'], subject='pid', correction=True, effsize='n2').round(3))
#     print(pg.mixed_anova(data=modified_df, dv='AUC', within='dim', between='method', subject='pid', correction=True, effsize='n2').round(3))
#     print(pg.pairwise_tests(data=modified_df, dv='AUC', within='method', subject='pid', padjust='bonf', effsize='eta-square').round(3))



#Simplicity -- AUC
def get_auc(m):
    return np.trapz(np.fromstring(m[1:-1], sep=' '), dx=1/100)

df = pd.read_csv('linear_results.csv')
df = pd.read_csv('linear_pretrained_results.csv')
df['AUC'] = df['m'].apply(get_auc)

print(df.groupby(['modality', 'dim_embedding'])['AUC'].mean().round(3))


# for modality in [ 'visual', 'auditory', 'kinetic']:
#     stats_df = df.query(f'modality == "{modality}" and dim_embedding == 8')

#     print(f'\n\n{modality}\n\n')
    
#     # print(pg.homoscedasticity(data=stats_df, dv='AUC', group='method'))
#     # print(stats_df.welch_anova(dv='AUC', between='method').round(3))
#     # print(pg.pairwise_gameshowell(data=stats_df, dv='AUC',
#     #                     between='method').round(3))
    
#     print(pg.rm_anova(data=stats_df, dv='AUC', within='method', subject='pid', correction=True, effsize='n2').round(3))
#     print(pg.pairwise_tests(data=stats_df, dv='AUC', within='method', subject='pid', padjust='bonf', effsize='eta-square').round(3))
