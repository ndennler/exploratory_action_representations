import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def show_visual():
    all_data = pd.read_csv('../data/all_data.csv')
    final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
    TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


    embeds = np.load('../data/embeds/visual&taskconditioned&clip_embeds&contrastive&all_signals&128.npy')

    example = final_signals.iloc[0]
    vis_id, signal = example['visual'], example['signal']

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


def show_auditory(method='contrastive'):
    all_data = pd.read_csv('../data/all_data.csv')
    final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
    TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


    embeds = np.load(f'../data/embeds/auditory&taskconditioned&ast_embeds&{method}&all_signals&128.npy')

    example = final_signals.iloc[0]
    aud_id, signal = example['auditory'], example['signal']

    aud = embeds[aud_id, TASK_INDEX_MAPPING[signal]]

    alignment = (aud @ embeds[:, TASK_INDEX_MAPPING[signal]].T)
    norms = np.linalg.norm(embeds[:, TASK_INDEX_MAPPING[signal]], axis=1)

    sorted_ims = np.argsort(alignment / norms)
    bottom, top = sorted_ims[:16], sorted_ims[-16:]


    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle('Least Similar ' + signal)
    print('least similar')
    axs = axs.flatten()
    for img, ax in zip(bottom, axs):
        im_path = all_data.query(f'type == "Audio" and id == {img}')['file']
        if len(im_path) > 0:
            im_path =im_path.values[0][:-4]
            print(im_path)
            img = Image.open("../data/auditory/aud/" + im_path + '.jpg')
            ax.imshow(img)
            ax.axis('off')

    print('\nmost similar')
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle('Most Similar ' + signal)
    axs = axs.flatten()
    for img, ax in zip(top, axs):
        im_path = all_data.query(f'type == "Audio" and id == {img}')['file']
        if len(im_path) > 0:
            im_path =im_path.values[0][:-4]
            print(im_path)
            img = Image.open("../data/auditory/aud/" + im_path + '.jpg')
            ax.imshow(img)
            ax.axis('off')


    print('\nstimulus')
    _, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs = axs.flatten()
    for img, ax in zip([aud_id], axs):
        im_path = all_data.query(f'type == "Audio" and id == {img}')['file'].values[0][:-4]
        print(im_path)
        img = Image.open("../data/auditory/aud/" + im_path + '.jpg')
        ax.imshow(img)
        ax.title.set_text(im_path)
        ax.axis('off')

    plt.show()


def show_kinetic(method='contrastive'):
    all_data = pd.read_csv('../data/all_data.csv')
    final_signals = pd.read_csv('../data/evaluation/concatenated_final_signals.csv')
    TASK_INDEX_MAPPING = {'idle': 0, 'searching': 1, 'has_information': 2, 'has_item': 3}


    embeds = np.load(f'../data/embeds/kinesthetic&taskconditioned&AE_embeds&{method}&all_signals&128.npy')

    example = final_signals.iloc[0]
    kin_id, signal = example['kinesthetic'], example['signal']

    kin = embeds[kin_id, TASK_INDEX_MAPPING[signal]]

    alignment = (kin @ embeds[:, TASK_INDEX_MAPPING[signal]].T)
    norms = np.linalg.norm(embeds[:, TASK_INDEX_MAPPING[signal]], axis=1)

    sorted_ims = np.argsort(alignment / norms)
    bottom, top = sorted_ims[:16], sorted_ims[-16:]


    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle('Least Similar ' + signal)
    print('least similar')
    axs = axs.flatten()
    for id, ax in zip(bottom, axs):
        im_path = all_data.query(f'type == "Movement" and id == {id}')['file']
        if len(im_path) > 0:
            im_path =im_path.values[0][:-4]
            print(im_path)
            img = Image.open(f"../data/kinetic/kin/{id}.png")
            ax.imshow(img)
            ax.axis('off')

    print('\nmost similar')
    fig, axs = plt.subplots(4, 4, figsize=(6, 6))
    fig.suptitle('Most Similar ' + signal)
    axs = axs.flatten()
    for id, ax in zip(top, axs):
        im_path = all_data.query(f'type == "Movement" and id == {id}')['file']
        if len(im_path) > 0:
            im_path =im_path.values[0][:-4]
            print(im_path)
            img = Image.open(f"../data/kinetic/kin/{id}.png")
            ax.imshow(img)
            ax.axis('off')


    print('\nstimulus')
    _, axs = plt.subplots(2, 2, figsize=(6, 6))
    axs = axs.flatten()
    for id, ax in zip([kin_id], axs):
        # im_path = all_data.query(f'type == "Movement" and id == {img}')['file'].values[0][:-4]
        # print(im_path)
        img = Image.open(f"../data/kinetic/kin/{id}.png")
        ax.imshow(img)
        ax.title.set_text(im_path)
        ax.axis('off')

    plt.show()

if __name__ == '__main__':
    show_kinetic()