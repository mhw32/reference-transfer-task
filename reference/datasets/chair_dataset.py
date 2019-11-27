from __future__ import print_function

import os
import json
from copy import deepcopy
from tqdm import tqdm
import numpy as np
import pandas as pd
from PIL import Image
from utils import OrderedCounter
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms
from collections import defaultdict

FILE_DIR = os.path.realpath(os.path.dirname(__file__))
RAW_DIR = os.path.join(FILE_DIR, '/mnt/fs5/wumike/datasets/chair2k_old/chairs2k')
NUMPY_DIR = os.path.join(RAW_DIR, 'numpy')

SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
PAD_TOKEN = '<pad>'
UNK_TOKEN = '<unk>'
TRAINING_PERCENTAGE = 80 / 100
TESTING_PERCENTAGE = 10 / 100
MIN_USED = 2
MAX_LEN = 15

class Chairs_ReferenceGame(data.Dataset):
    def __init__(self, vocab=None, split='Train', context_condition='far', 
                 hard=False, image_size=32, image_transform=None):
        super(Chairs_ReferenceGame, self).__init__()

        self.images = np.load(os.path.join(NUMPY_DIR, 'images.npy'))
        self.context_condition = context_condition
        self.hard = hard
        self.split = split
        assert self.split in ('Train', 'Validation', 'Test')
       
        self.names = np.load(os.path.join(NUMPY_DIR, 'names.npy'))
        chair_list = []
        for i in self.names:
            i = str(i.decode('utf-8'))
            chair_list.append(i)
        self.names  = chair_list

        npy_path = os.path.join(RAW_DIR, 'cleaned_data_{}.npy'.format(context_condition))
        # print('loading CSV file ...')
        csv_path = os.path.join(RAW_DIR, 'chairs2k_group_data.csv')
        df = pd.read_csv(csv_path)
        df = df[df['correct'] == True]
        df = df[df['communication_role'] == 'speaker']
        
        self.random_state = np.random.RandomState(120)

        # split by train / validation / test
        split_indices = np.arange(len(df))
        train_len = int(TRAINING_PERCENTAGE * len(split_indices))
        test_len = int(TESTING_PERCENTAGE * len(split_indices))
        
        if self.split == 'Train':
            split_indices = split_indices[:train_len]
        if self.split == 'Validation':
            split_indices = split_indices[train_len:-test_len]
        if self.split == 'Test':
            split_indices = split_indices[-test_len:]
        df = df.reindex(split_indices)

        # split by context condition
        if context_condition != 'all':
            assert context_condition in ['far', 'close', 'split']
            df = df[df['context_condition'] == context_condition]
        
        # note that target_chair is always the chair 
        # so label is always 3
        df = df[['chair_a', 'chair_b', 'chair_c', 'target_chair', 'text']]
        df = df.dropna()
        data = np.asarray(df)
        data = self.clean_data(data, self.names)

        # replace target_chair with a label
        labels = []
        for i in range(len(data)):
            if data[i, 3] == data[i, 0]:
                labels.append(0)
            elif data[i, 3] == data[i, 1]:
                labels.append(1)
            elif data[i, 3] == data[i, 2]:
                labels.append(2)
            else:
                raise Exception('bad label')
        labels = np.array(labels)

        self.data = data
        self.labels = labels

        text = [d[-1] for d in data]
    
        if vocab is None:
            print('\nbuilding vocab ...')
            self.vocab = self.build_vocab(text)
        else:
            self.vocab = vocab
        
        self.w2i, self.i2w = self.vocab['w2i'], self.vocab['i2w']
        self.vocab_size = len(self.w2i)

        self.sos_token = SOS_TOKEN
        self.eos_token = EOS_TOKEN
        self.pad_token = PAD_TOKEN
        self.unk_token = UNK_TOKEN

        self.sos_index = self.w2i[self.sos_token]
        self.eos_index = self.w2i[self.eos_token]
        self.pad_index = self.w2i[self.pad_token]
        self.unk_index = self.w2i[self.unk_token]

        self.inputs, self.lengths, self.max_length = self.process_texts(text)

        self.image_transform = image_transform

        print("vocabulary size: {}".format(self.vocab_size))
        print("{} dataset preparation complete.\n".format(split))

        # print(self.vocab)

    def build_vocab(self, texts):
        w2c = defaultdict(int)
        i2w, w2i = {}, {}
        for text in texts:
            tokens = preprocess_text_chairs(text)
            for token in tokens:
                w2c[token] += 1
        indexCount = 0
        for token in w2c.keys():
            if w2c[token] >= MIN_USED:
                w2i[token] = indexCount
                i2w[indexCount] = token
                indexCount += 1
        w2i[SOS_TOKEN] = indexCount
        w2i[EOS_TOKEN] = indexCount+1
        w2i[UNK_TOKEN] = indexCount+2
        w2i[PAD_TOKEN] = indexCount+3
        i2w[indexCount] = SOS_TOKEN
        i2w[indexCount+1] = EOS_TOKEN
        i2w[indexCount+2] = UNK_TOKEN
        i2w[indexCount+3] = PAD_TOKEN

        vocab = {'i2w': i2w, 'w2i': w2i}

        print("total number of words used at least twice: %d" % len(w2i))
        print("total number of different words: %d" % len(w2c.keys()))
        return vocab

    def clean_data(self, data, names):
        new_data = []
        for i in tqdm(range(len(data))):
            chair_a, chair_b, chair_c, chair_target, _ = data[i]
            if chair_a + '.png' not in names:
                continue
            if chair_b + '.png' not in names:
                continue
            if chair_c + '.png' not in names:
                continue
            if chair_target + '.png' not in names:
                continue
            new_data.append(data[i])
        new_data = np.array(new_data)
        return new_data

    def process_texts(self, texts):
        sources, lengths = [], []

        n = len(texts)
        for i in range(n):
            tokens = preprocess_text_chairs(texts[i])
            tokens = [SOS_TOKEN] + tokens
            # + [EOS_TOKEN]
            length = len(tokens)
            if length < MAX_LEN:
                tokens.extend([PAD_TOKEN] * (MAX_LEN - length))
            else:
                tokens = tokens[:MAX_LEN]
                length = MAX_LEN
            indices = [self.vocab['w2i'].get(token, self.vocab['w2i'][UNK_TOKEN]) for token in tokens]

            sources.append(np.array(indices))
            lengths.append(length)
        
        sources = np.array(sources)
        lengths = np.array(lengths)
        return sources, lengths, MAX_LEN

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        label = self.labels[index]

        chair_a, chair_b, chair_c, _, _ = self.data[index]
        
        chair_a = chair_a + '.png'
        chair_b = chair_b + '.png'
        chair_c = chair_c + '.png'

        chair_names = list(self.names)
        index_a = chair_names.index(chair_a)
        index_b = chair_names.index(chair_b)
        index_c = chair_names.index(chair_c)

        chair_a_np = self.images[index_a][0]
        chair_b_np = self.images[index_b][0]
        chair_c_np = self.images[index_c][0]

        chair_a_pt = torch.from_numpy(chair_a_np).unsqueeze(0)
        chair_a = transforms.ToPILImage()(chair_a_pt).convert('RGB')

        chair_b_pt = torch.from_numpy(chair_b_np).unsqueeze(0)
        chair_b = transforms.ToPILImage()(chair_b_pt).convert('RGB')

        chair_c_pt = torch.from_numpy(chair_c_np).unsqueeze(0)
        chair_c = transforms.ToPILImage()(chair_c_pt).convert('RGB')

        if self.image_transform is not None:
            chair_a = self.image_transform(chair_a)
            chair_b = self.image_transform(chair_b)
            chair_c = self.image_transform(chair_c)

        inputs = self.inputs[index]
        length = self.lengths[index]
        trans = transforms.ToTensor()

        inputs = torch.from_numpy(inputs).long()

        return index, trans(chair_a), trans(chair_b), trans(chair_c), inputs, length, label


def preprocess_text_chairs(text):
    text = text.lower() 
    tokens = word_tokenize(text)
    return tokens
