import re
import os
import random
import shutil
import librosa
import librosa.display
import pandas as pd

def get_label(filename):
    m = re.search('(.+?)_', filename)
    if m:
        return m.group(1)
    return None

def delete_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)

def delete_dirs():
    delete_dir("data/dev")
    delete_dir("data/train")
    delete_dir("data/test")

def create_dirs():
    os.mkdir("data/dev")
    os.mkdir("data/train")
    os.mkdir("data/test")

def get_speaker_file_dictionary():
    prepared_files = {}
    for root, dirs, files in os.walk("data/raw"):
        for filename in files:
            m = re.search('(.+?)_', filename)
            if m:
                found = m.group(1)
                if found not in prepared_files:
                    prepared_files[found] = []
                prepared_files[found].append(filename)
    return prepared_files

def make_data_for(list, mode):
    for file in list:
        shutil.copyfile(f"data/raw/{file}",f"data/{mode}/{file}")

def get_mode_counts(size):
    n_test = int(size // 10 * 1.5)
    n_train = int(size // 10 * 8)
    n_dev = size - (n_test + n_train)
    return n_dev, n_train, n_test

def process_file_list(files, n_dev, n_train, n_test):
    make_data_for(files[0:n_dev], "dev")
    make_data_for(files[n_dev: n_dev+n_train], "train")
    make_data_for(files[n_dev+n_train : ], "test")

def prepare():
    delete_dirs()
    create_dirs()
    prepared_files = get_speaker_file_dictionary()

    for key,value in prepared_files.items():
        n_dev, n_train, n_test = get_mode_counts(len(value))
        random.shuffle(value)
        process_file_list(value, n_dev, n_train, n_test )


def stat():
    print(f"Total dev files: {len(os.listdir('data/dev'))}")
    print(f"Total train files: {len(os.listdir('data/train'))}")
    print(f"Total test files: {len(os.listdir('data/test'))}")


def prepare_if_needed():
    if not os.path.exists("data/dev"):
        prepare()
    #stat()

def get_mfcc_data(mode):
    df = pd.DataFrame(columns=['feature'])

    # loop feature extraction over the entire dataset
    counter = 0
    labels = []
    for index, path in enumerate(os.listdir(f"data/{mode}")):
        X, sample_rate = librosa.load(f"data/{mode}/{path}"
                                      , res_type='kaiser_fast'
                                      , duration=2.5
                                      , sr=44100
                                      , offset=0.5
                                      )
        sample_rate = torch.array(sample_rate)

        # mean as the feature. Could do min and max etc as well.
        mfccs = torch.mean(librosa.feature.mfcc(y=X,
                                             sr=sample_rate,
                                             n_mfcc=13),
                        axis=0)
        df.loc[counter] = [mfccs]
        labels.append(get_label(path))
        counter = counter + 1

    # Check a few records to make sure its processed successfully
    # print(len(df))
    a_data_frame = pd.DataFrame()
    a_data_frame = pd.concat([a_data_frame, pd.DataFrame(df['feature'].values.tolist())], axis=1)
    a_data_frame = a_data_frame.fillna(0)
    return a_data_frame.values, labels

def get_mel_data(mode):
    counter = 0
    labels = []
    mels = []
    for index, path in enumerate(os.listdir(f"data/{mode}")):
        X, sample_rate = librosa.load(f"data/{mode}/{path}"
                                      , res_type='kaiser_fast'
                                      , duration=2
                                      , sr=44100
                                      , offset=0.5
                                      )
        X, _ = librosa.effects.trim(X)
        mel_spect = librosa.feature.melspectrogram(y=X, sr=sample_rate, n_fft=2048, hop_length=1024)
        mel_spect = librosa.amplitude_to_db(mel_spect, ref=torch.max)

        mels.append(mel_spect)
        labels.append(get_label(path))
        counter = counter + 1

    new_mels = []
    new_labels = []
    for mel, label in zip(mels, labels):
        mel = trim_mel(mel)
        if mel is not None:
            new_mels.append(mel)
            new_labels.append(label)

    return new_mels, new_labels

def trim_mel(mel):
    expected_height = 128
    expected_width = 80
    arr = torch.zeros((128, 80))
    if len(mel) < expected_height:
        return None

    for i in range(expected_height):
        if len(mel[i]) < expected_width:
            return None
        for j in range(expected_width):
            if mel[i][j] is None:
                return None
            arr[i,j] = mel[i][j]
    return arr
