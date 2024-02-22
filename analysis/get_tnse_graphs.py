import pandas as pd
import numpy as np
import sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import plotly.express as px

# refactor to use sysargs to take command line arguments later
if __name__ == '__main__':
    if len(sys.argv) != 5:
        print("Not valid input")
        exit()
    TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}
    signal = sys.argv[1]
    modality = sys.argv[2]
    type = sys.argv[3]
    file_name = sys.argv[4]
    all_data_csv = pd.read_csv('../data/all_data.csv')
    final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
    
    # get filtered version of each of the data
    # change
    # use Video, Audio, or Movement
    all_data_filtered = all_data_csv[all_data_csv['type'] == type][:]
    print(all_data_filtered['id'].max())

    #change
    embeddings = np.load(f'../data/embeds/{file_name}')
    flattened_embeddings = embeddings[:,TASK_INDEX_MAPPING[signal],:]
    print(embeddings.shape)

    # get final signal embeddings
    # change
    # use visual,auditory, or kinesthetic
    selected_id = final_signals.query(f"signal=='{signal}'")[modality].values
    selected_id = selected_id[selected_id > 0] 
    # first flatten with pca to 50 dimensions
    embeddings_pca = PCA(n_components = 50).fit_transform(flattened_embeddings)
    # fit tsne
    tsne = TSNE(n_components=2, verbose=1, perplexity=80, n_iter=300)
    embeddings_2d = tsne.fit_transform(embeddings_pca)
    selected_embeddings = embeddings_2d[selected_id]

    plot_data = []
    for i, row in all_data_filtered.iterrows():
        id = row['id']
        tag = row['tags']
        x = embeddings_2d[int(id), 0]
        y = embeddings_2d[int(id), 1]
        color = "red" if int(id) in selected_id else "blue" 
        size = 2 if int(id) in selected_id else .1     
        plot_data.append({"id": id, "tag": tag, "x": x, "y": y, "color": color, "size": size})
    plot_df = pd.DataFrame(plot_data)
    print(plot_df['color'].unique())
    fig = px.scatter(plot_df, x = "x", y = "y", color = "color", size = "size" , hover_data = ["tag", "id"], opacity = .5)
    fig.show()
