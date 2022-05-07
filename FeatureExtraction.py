import pandas as pd
import numpy as np

import os
import sys

# librosa is a Python library for analyzing audio and music. It can be used to extract the data from the audio files we will see it later.
import librosa
import librosa.display as ld
import matplotlib.pyplot as plt

# Crema path
Crema = "E:\\Users\\Raven\\Documents\\Datasets\\CREMA-D\\CREMA-D\\Audio\\AudioWAV - Organised - Copy (2)\\"

# Folder paths
crema_folder_list = os.listdir(Crema)

# Preparing Crema_df
file_emotion = []
file_path = []

for i in range(0, 3):
    crema_directory_list = os.listdir(Crema + crema_folder_list[i])

    for file in crema_directory_list:
        # storing file paths
        file_path.append(Crema + crema_folder_list[i] + "\\" + file)
        # storing file emotions
        part = [x.strip() for x in file.split('-')]

        if part[0] == 'Sad':
            file_emotion.append('sad')
        elif part[0] == 'Anger':
            file_emotion.append('angry')
        elif part[0] == 'Disgust':
            file_emotion.append('disgust')
        elif part[0] == 'Fear':
            file_emotion.append('fear')
        elif part[0] == 'Happy':
            file_emotion.append('happy')
        elif part[0] == 'Neutral':
            file_emotion.append('neutral')
        else:
            file_emotion.append('Unknown')

# dataframe for emotion of files
emotion_df = pd.DataFrame(file_emotion, columns=['Emotions'])

# dataframe for path of files.
path_df = pd.DataFrame(file_path, columns=['Path'])
Crema_df = pd.concat([emotion_df, path_df], axis=1)
Crema_df.head()

# creating Dataframe using all the 4 dataframes we created so far.
data_path = pd.concat([Crema_df], axis = 0)
data_path.to_csv("data_path.csv", index=False)
data_path.head()

# Extracting data and sample_rate
path = np.array(data_path.Path)[0]
data, sample_rate = librosa.load(path)

# Functions
def getEmotion(path):
    part = path.split('_')
    if part[2] == 'SAD':
        return "sad"
    elif part[2] == 'ANG':
        return "angry"
    elif part[2] == 'DIS':
        return "disgust"
    elif part[2] == 'FEA':
        return "fear"
    elif part[2] == 'HAP':
        return "happy"
    elif part[2] == 'NEU':
        return "neutral"
    else:
        return "unknown"

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(data, rate)

def shift(data):
    shift_range = int(np.random.uniform(low=-5, high = 5)*1000)
    return np.roll(data, shift_range)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

# Feature extraction functions
def extract_features(data):
    # ZCR
    result = np.array([])
    zcr = np.mean(librosa.feature.zero_crossing_rate(y=data).T, axis=0)
    result = np.hstack((result, zcr))  # stacking horizontally

    # Chroma_stft
    stft = np.abs(librosa.stft(data))
    chroma_stft = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
    result = np.hstack((result, chroma_stft))  # stacking horizontally

    # MFCC
    mfcc = np.mean(librosa.feature.mfcc(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mfcc))  # stacking horizontally

    # Root Mean Square Value
    rms = np.mean(librosa.feature.rms(y=data).T, axis=0)
    result = np.hstack((result, rms))  # stacking horizontally

    # MelSpectogram
    mel = np.mean(librosa.feature.melspectrogram(y=data, sr=sample_rate).T, axis=0)
    result = np.hstack((result, mel))  # stacking horizontally

    return result


def get_features(path):
    # duration and offset are used to take care of the no audio in start and the ending of each audio files as seen above.
    data, sample_rate = librosa.load(path, duration=2.5, offset=0.6)

    # without augmentation
    res1 = extract_features(data)
    result = np.array(res1)

    # data with noise
    noise_data = noise(data)
    res2 = extract_features(noise_data)
    result = np.vstack((result, res2))  # stacking vertically

    # data with stretching and pitching
    new_data = stretch(data)
    data_stretch_pitch = pitch(new_data, sample_rate)
    res3 = extract_features(data_stretch_pitch)
    result = np.vstack((result, res3))  # stacking vertically

    return result

X, Y = [], []
for path, emotion in zip(data_path.Path, data_path.Emotions):
    feature = get_features(path)
    for ele in feature:
        X.append(ele)
        # appending emotion 3 times as we have made 3 augmentation techniques on each audio file.
        Y.append(emotion)

len(X), len(Y), data_path.Path.shape

Features = pd.DataFrame(X)
Features['labels'] = Y
Features.to_csv("C:\\Users\\Raven\\Desktop\\featuresSingle.csv", index=False)