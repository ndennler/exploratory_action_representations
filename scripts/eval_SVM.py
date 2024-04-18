import sklearn
from sklearn.model_selection import train_test_split, StratifiedKFold
import pandas as pd
import numpy as np
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from sklearn.metrics import f1_score

final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
all_data = pd.read_csv('../data/all_data.csv')

TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}
MODALITY_TYPE_MAPPING = {'visual': 'Video', 'auditory': 'Audio', 'kinesthetic': 'Movement'}
MODALITY_PRETRAINED_MAPPING = {'visual': 'clip_embeds', 'auditory': 'ast_embeds', 'kinesthetic': 'AE_embeds'}

scores = []

for EMBED_TYPE in ['contrastive', 'autoencoder', 'random', 'VAE', 'contrastive+autoencoder']:
    for modality in tqdm(['visual', 'auditory', 'kinesthetic']):
        for signal in final_signals['signal'].unique():

            #get the indices that were ultimately selected as the final signals (positive examples)
            final_indices = final_signals.query(f'signal == "{signal}"')[modality].values
            final_indices = final_indices[final_indices > 0]

            #get the rest of the indices (negative examples)
            indices = all_data.query(f'type == "{MODALITY_TYPE_MAPPING[modality]}"')['id'].values 
            indices = indices[~np.in1d(indices,final_indices)]
            
            embeds = np.load(f'../data/embeds/{modality}&taskconditioned&raw&{EMBED_TYPE}&all_signals&128.npy')

            X = np.append(final_indices, indices)
            X = embeds[X, TASK_INDEX_MAPPING[signal], :]
            y = np.array([1]*len(final_indices) + [0]*len(indices))

            skf = StratifiedKFold(n_splits=5)

            for train, test in skf.split(X, y):

                X_train, X_test = X[train], X[test]
                y_train, y_test = y[train], y[test]

        
                from sklearn.svm import SVC
                clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
                clf.fit(X_train, y_train)

                y_preds = clf.predict_proba(X_test)

                # score = f1_score(y_test, y_preds, average='weighted')
                score = y_preds[y_test[y_test > 0], 1].mean()

                scores.append({
                    'modality': modality,
                    'signal': signal,
                    'embed_type': EMBED_TYPE,
                    'score': score
                })

scores = pd.DataFrame(scores)
scores.to_csv(f'../data/results/SVM_results.csv', index=False)

print(scores.groupby(['modality', 'embed_type'])['score'].mean())
print(scores.groupby(['modality', 'embed_type'])['score'].std())
            