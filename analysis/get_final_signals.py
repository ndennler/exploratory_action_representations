import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
all_data = pd.read_csv('../data/all_data.csv')

SIGNAL = 'has_information'
MODALITY = 'visual'

final_signals = final_signals.query('signal == @SIGNAL')[MODALITY].values

fig, axs = plt.subplots(5, 5, figsize=(6, 6))
fig.suptitle('SELECTED ' + SIGNAL)
axs = axs.flatten()

if MODALITY == 'visual':
    for i, id in enumerate(final_signals):
        im_path = all_data.query(f'type == "Video" and id == {id}')['file']
        if len(im_path) > 0:
            im_path =im_path.values[0][:-4]
            print(im_path)
            img = Image.open("../data/visual/vis/" + im_path + '.jpg')
            axs[i].imshow(img)
            axs[i].axis('off')

plt.show()