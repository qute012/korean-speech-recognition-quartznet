"""Data loader for the LibriSpeech dataset. See: http://www.openslr.org/12/"""
__author__ = 'Erdene-Ochir Tuguldur'

import os

import numpy as np
from torch.utils.data import Dataset
import pickle

#with open('data/aihub/char2id.pkl', 'rb') as f:
#    char2id = pickle.load(f)
    
with open('/home/cilab/LabMembers/DJ/final_exp/QUARTZNET/datasets/meta/aihub/id2char.pkl', 'rb') as f:
    vocab = ''.join(pickle.load(f)[:991])

with open('/home/cilab/LabMembers/DJ/final_exp/QUARTZNET/datasets/meta/aihub/train_data.pkl', 'rb') as f:
    train_data = pickle.load(f)
    
with open('/home/cilab/LabMembers/DJ/final_exp/QUARTZNET/datasets/meta/aihub/test_data.pkl', 'rb') as f:
    test_data = pickle.load(f)
    
#PAD_TOKEN = int(char2id['_'])
train_data = train_data[:int(len(train_data)*0.2)]
vocab += 'U'
#vocab = ''.join(id2char)
char2idx = {char: idx for idx, char in enumerate(vocab)}
idx2char = {idx: char for idx, char in enumerate(vocab)}


def convert_text(text):
    return [char2idx[char] for char in text if char != '_']

"""
def read_metadata(dataset_path, metadata_file, max_duration):
    fnames, texts = [], []

    reader = csv.reader(open(metadata_file, 'rt'))
    for line in reader:
        fname, duration, text = line[0], line[1], line[2]
        if fname.endswith('0.9.wav') or fname.endswith('1.1.wav'):
            continue
        try:
            duration = float(duration)
            if duration > max_duration:
                continue
        except ValueError:
            continue
        fnames.append(os.path.join(dataset_path, fname))
        texts.append(np.array(convert_text(text)))

    return fnames, texts
"""

class AihubSpeech(Dataset):

    def __init__(self, name='train', max_duration=None, transform=None):
        self.transform = transform

        if name=='train':
            self.pair_data = train_data
        else:
            self.pair_data = test_data

    def __getitem__(self, index):
        fname, text = self.pair_data[index]
        data = {
            'fname': fname,
            'text': [char2id.get(char) if char2id.get(char)!=None else char2id.get('U') for char in list(text)]
        }

        if self.transform is not None:
            data = self.transform(data)

        return data

    def __len__(self):
        return len(self.pair_data)