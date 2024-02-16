import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

data = pd.read_csv('../data/all_data.csv')

# visual_concepts = {
#     0 : ['kid', 'baby', 'toy'],
#     1 : ['baggage', 'hand', 'hands', 'luggage', 'box', 'package', 'parcel', 'bag'],
#     2 : ['happy', 'smiling', 'smile', 'happiness',  'smiley'],
#     3 : ['time', 'clock', 'wait', 'hour', 'day'],
#     4 : ['sleep', 'relax', 'rest', 'eye'],
#     5 : ['communications', 'message', 'chat', 'speech', 'bubble', 'conversation', 'talk'],
#     6 : ['idea', 'lightbulb', 'information', 'info', 'brain', 'illumination'],
#     7 : ['maps', 'map', 'location', 'arrow', 'pin', 'sign', 'transportation', 'vehicle', 'navigation', 'cursor', 'pointer'],
#     8 : ['business', 'finance', 'card', 'payment', 'finance', 'banking'],
#     9 : ['magnifying', 'glass', 'search', 'eye', 'radar', 'glasses'],
#     10 : ['animal' , 'pet', 'animals'],
# }

# vis_data = data.query('type == "Video"')
# handcrafted_features = np.zeros((vis_data['id'].max() + 1, 11))

# # for each row, create a label vector of length 100
# for i, row in vis_data.iterrows():
#     tags = row['tags'].split(',')

#     label = np.zeros(11)

#     for i in range(11):
#         value = 0
#         for tag in visual_concepts[i]:
#             if tag in tags:
#                 value += 1
        
#         label[i] = value

#     print(label)


#     handcrafted_features[row['id']] = label

# print(handcrafted_features)

# np.save('../data/handcrafted_features/visual.npy', handcrafted_features)


'''
Handcrafted features for video come from the tags associated with each video.
To do this we will get the top 99 tags and use the 100th tag as the "other" tag.
'''
def generate_visual_features():
    vis_data = data.query('type == "Video"')
    tags = Counter()

    for tag in vis_data['tags']:
        tag = tag.split(',')
        for t in tag:
            tags.update([t])

    top_tags = [t[0] for t in tags.most_common(100)[1:]] #see analysis/get_most_common_vis_tags.py for specifics on why this number was selected.

    handcrafted_features = np.zeros((vis_data['id'].max() + 1, 100))

    # for each row, create a label vector of length 100
    for i, row in vis_data.iterrows():
        tags = row['tags'].split(',')
        label = np.zeros(100)
        for i,tag in enumerate(top_tags):
            if tag in tags:
                label[i] = 1
        if label.sum() == 0:
            label[-1] = 1

        handcrafted_features[row['id']] = label

    np.save('../data/handcrafted_features/visual.npy', handcrafted_features)
    print('features per category:')
    print(np.sum(handcrafted_features, axis=0))


'''
Handcrafted Features for sound come from librosa's feature extraction.
We will use the following features:
    - Spectral Centroid
    - Spectral Bandwidth
    - Spectral Contrast
    - Spectral Rolloff
    - Mel-Frequency Cepstral Coefficients (MFCC)
    - Chroma Frequencies
    - Tonnetz
'''
from librosa.feature import spectral_centroid, spectral_bandwidth, spectral_contrast, spectral_rolloff, mfcc, chroma_stft, tonnetz
from librosa import load
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

def generate_audio_features():
    aud_data = data.query('type == "Audio"')

    handcrafted_features = np.zeros((aud_data['id'].max() + 1, 96)) #96 is the number of features we are extracting

    for i, row in tqdm(aud_data.iterrows()):
        audio, sr = load('../data/auditory/aud/' + row['file'])
        audio = np.ma.masked_equal(audio, 0)

        centroid = spectral_centroid(y=audio, sr=sr)
        bandwidth = spectral_bandwidth(y=audio, sr=sr)
        contrast = spectral_contrast(y=audio, sr=sr)
        rolloff = spectral_rolloff(y=audio, sr=sr)
        mfccs = mfcc(y=audio, sr=sr)
        chroma = chroma_stft(y=audio, sr=sr)
        tonnetzs = tonnetz(y=audio, sr=sr)

        features = np.concatenate((centroid, bandwidth, contrast, rolloff, mfccs, chroma, tonnetzs), axis=0)
        label = np.concatenate((np.median(features, axis=1), np.std(features, axis=1)), axis=0)
        
        handcrafted_features[row['id']] = label

    scaler = StandardScaler()
    handcrafted_features = scaler.fit_transform(handcrafted_features)

    np.save('../data/handcrafted_features/auditory.npy', handcrafted_features)

'''
Handcrafted Features for movements come from the following characteristics:
- max, min, mean, median, std for each degree of freedom
- max, min, mean, median, std for each degree of freedom's velocity
- number of peaks for each degree of freedom
'''
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler

def generate_movement_features():
    kin_data = data.query('type == "Movement"')
    trajectories = np.load('../data/kinetic/behaviors.npy')

    handcrafted_features = np.zeros((kin_data['id'].max() + 1, 33)) #33 is the number of features we are extracting

    for i, row in tqdm(kin_data.iterrows()):
        kin = trajectories[row['id']]
        derivative = np.diff(kin, axis=0)
        
        min = np.min(kin, axis=0)
        max = np.max(kin, axis=0)
        mean = np.mean(kin, axis=0)
        median = np.median(kin, axis=0)
        std = np.std(kin, axis=0)
        min_vel = np.min(derivative, axis=0)
        max_vel = np.max(derivative, axis=0)
        mean_vel = np.mean(derivative, axis=0)
        median_vel = np.median(derivative, axis=0)
        std_vel = np.std(derivative, axis=0)
        
        num_peaks = np.zeros((3))
        for j in range(3):
            num_peaks[j] = len(np.where(np.diff(np.sign(derivative[:,j])))[0])

        label = np.concatenate((min, max, mean, median, std, min_vel, max_vel, mean_vel, median_vel, std_vel, num_peaks), axis=0)
        handcrafted_features[row['id']] = label

    scaler = StandardScaler()
    handcrafted_features = scaler.fit_transform(handcrafted_features)

    np.save('../data/handcrafted_features/kinetic.npy', handcrafted_features)   

from PIL import Image
import requests

from transformers import CLIPProcessor, CLIPModel

def generate_visual_clip_features():
    vis_data = data.query('type == "Video"')

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    image = Image.open(requests.get(url, stream=True).raw)

    descriptions = [
        'a photo of an object that can hold something in it.',
        'a photo of an object that can be used to see things',
        'a photo of an object that can move',
        'a symbol that represents information',
        'a symbol that represents time',
        'a symbol that represents a location',
    ]

    handcrafted_features = np.zeros((vis_data['id'].max() + 1, len(descriptions)))

    # for each row, create a label vector of length 100
    for i, row in tqdm(vis_data.iterrows()):
        fname =  '../data/visual/vis/' + row['file'].split('.')[0] + '.jpg'
        image = Image.open(fname)
        inputs = processor(text=descriptions, images=image, return_tensors="pt", padding=True)

        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image  # this is the image-text similarity score

        handcrafted_features[row['id']] = logits_per_image.detach().numpy()[0]
        if i % 100 == 0:
            print(np.std(handcrafted_features, axis=0))
        

    np.save('../data/handcrafted_features/visual.npy', handcrafted_features)
    print('features per category:')
    print(np.sum(handcrafted_features, axis=0))

    

if __name__ == '__main__':
    # generate_visual_features()
    # generate_audio_features()
    # generate_movement_features()
    generate_visual_clip_features()