import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import av 
from tqdm import tqdm
import transformers
from transformers import AutoProcessor, AutoModel, EncodecModel, ASTModel
import torchaudio
import torch

def read_video_pyav(container, indices):
    '''
    Decode the video with PyAV decoder.
    Args:
        container (`av.container.input.InputContainer`): PyAV container.
        indices (`List[int]`): List of frame indices to decode.
    Returns:
        result (np.ndarray): np array of decoded frames of shape (num_frames, height, width, 3).
    '''
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def sample_frame_indices(clip_len, frame_sample_rate, seg_len):
    '''
    Sample a given number of frame indices from the video.
    Args:
        clip_len (`int`): Total number of frames to sample.
        frame_sample_rate (`int`): Sample every n-th frame.
        seg_len (`int`): Maximum allowed index of sample's last frame.
    Returns:
        indices (`List[int]`): List of sampled frame indices
    '''
    converted_len = int(clip_len * frame_sample_rate)
    end_idx = np.random.randint(converted_len, seg_len)
    start_idx = end_idx - converted_len
    indices = np.linspace(start_idx, end_idx, num=clip_len)
    indices = np.clip(indices, start_idx, end_idx - 1).astype(np.int64)
    return indices

def convert_audio(wav: torch.Tensor, sr: int, target_sr: int, target_channels: int):
    assert wav.dim() >= 2, "Audio tensor must have at least 2 dimensions"
    assert wav.shape[-2] in [1, 2], "Audio must be mono or stereo."
    *shape, channels, length = wav.shape
    if target_channels == 1:
        wav = wav.mean(-2, keepdim=True)
    elif target_channels == 2:
        wav = wav.expand(*shape, target_channels, length)
    elif channels == 1:
        wav = wav.expand(target_channels, -1)
    else:
        raise RuntimeError(f"Impossible to convert from {channels} to {target_channels}")
    wav = torchaudio.transforms.Resample(sr, target_sr)(wav)
    return wav

def generate_xclip_embeddings():
    all_videos = pd.read_csv('../data/all_data.csv').query('type=="Video"')
    xclip_embeds = np.zeros((all_videos['id'].max() + 1, 512)) #outputs 512 features

    for i in tqdm(range(len(all_videos))):
        row = all_videos.iloc[i]
        container = av.open('../data/visual/vis/' + row['file'])
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        print(video.shape)
        print()

        inputs = processor(videos=list(video), return_tensors="pt")

        video_features = model.get_video_features(**inputs)
        xclip_embeds[row['id']] = video_features.detach().numpy()[0]


    np.save('../data/visual/xclip_embeds.npy', xclip_embeds)

def generate_xclip_embeddings_movement():
    all_videos = pd.read_csv('../data/all_data.csv').query('type=="Movement"')
    xclip_embeds = np.zeros((all_videos['id'].max() + 1, 512)) #outputs 512 features

    for i in tqdm(range(len(all_videos))):
        row = all_videos.iloc[i]
        container = av.open('../data/kinetic/kin/' + str(row['id']) + '.mp4')
        indices = sample_frame_indices(clip_len=8, frame_sample_rate=1, seg_len=container.streams.video[0].frames)
        video = read_video_pyav(container, indices)

        processor = AutoProcessor.from_pretrained("microsoft/xclip-base-patch32")
        model = AutoModel.from_pretrained("microsoft/xclip-base-patch32")

        print(video.shape)
        print()

        inputs = processor(videos=list(video), return_tensors="pt")

        video_features = model.get_video_features(**inputs)
        xclip_embeds[row['id']] = video_features.detach().numpy()[0]


    np.save('../data/kinetic/xclip_embeds.npy', xclip_embeds)


def generate_AST_features():
    transformers.utils.logging.set_verbosity_error()
    
    all_sounds= pd.read_csv('../data/all_data.csv').query('type=="Audio"')
    AST_features = np.zeros((all_sounds['id'].max() + 1, 768)) #outputs 768 features

    for i in tqdm(range(len(all_sounds))):
        row = all_sounds.iloc[i]

        wav, sr = torchaudio.load('../data/auditory/aud/' + all_sounds.iloc[i]['file'])
        wav = convert_audio(wav, sr, 16_000, 1)

        processor = AutoProcessor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
        model = ASTModel.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")

        # audio file is decoded on the fly
        inputs = processor(wav[0], sampling_rate=16_000, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        last_hidden_states = outputs.pooler_output
        AST_features[row['id']] = last_hidden_states.detach().numpy()[0]
    
    np.save('../data/auditory/ast_embeds.npy', AST_features)

if __name__ == '__main__':
    generate_xclip_embeddings_movement()
    # generate_AST_features()