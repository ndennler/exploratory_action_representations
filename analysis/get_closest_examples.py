import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


all_data = pd.read_csv('../data/all_data.csv')
final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}

embeds = np.load('../data/embeds/visual&taskconditioned&raw&VAE&all_signals&64.npy')

example = final_signals.iloc[0]
vis_id, aud_id, kin_id, signal = example['visual'], example['auditory'], example['kinesthetic'], example['signal']

vis = embeds[vis_id, TASK_INDEX_MAPPING[signal]]

alignment = (vis @ embeds[:, TASK_INDEX_MAPPING[signal]].T)
norms = np.linalg.norm(embeds[:, TASK_INDEX_MAPPING[signal]], axis=1)

sorted_ims = np.argsort(alignment / norms)
bottom, top = sorted_ims[:16], sorted_ims[-19:-3]


fig, axs = plt.subplots(4, 4, figsize=(6, 6))
fig.suptitle('Least Similar ' + signal)
print('least similar')
axs = axs.flatten()
for img, ax in zip(bottom, axs):
    im_path = all_data.query(f'type == "Video" and id == {img}')['file']
    if len(im_path) > 0:
        im_path =im_path.values[0][:-4]
        print(im_path)
        img = Image.open("../data/visual/vis/" + im_path + '.jpg')
        ax.imshow(img)
        ax.axis('off')

print('most similar')
fig, axs = plt.subplots(4, 4, figsize=(6, 6))
fig.suptitle('Most Similar ' + signal)
axs = axs.flatten()
for img, ax in zip(top, axs):
    im_path = all_data.query(f'type == "Video" and id == {img}')['file']
    if len(im_path) > 0:
        im_path =im_path.values[0][:-4]
        print(im_path)
        img = Image.open("../data/visual/vis/" + im_path + '.jpg')
        ax.imshow(img)
        ax.axis('off')


print('stimulus')
_, axs = plt.subplots(2, 2, figsize=(6, 6))
axs = axs.flatten()
for img, ax in zip([vis_id], axs):
    im_path = all_data.query(f'type == "Video" and id == {img}')['file'].values[0][:-4]
    print(im_path)
    img = Image.open("../data/visual/vis/" + im_path + '.jpg')
    ax.imshow(img)
    ax.axis('off')

plt.show()
