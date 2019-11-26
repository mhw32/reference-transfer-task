import os
import json
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from nltk import sent_tokenize, word_tokenize

import torch
import torch.utils.data as data
from torchvision import transforms

from reference.text_utils import (
    SOS_TOKEN,
    EOS_TOKEN,
    PAD_TOKEN,
    UNK_TOKEN,
)
from reference.utils import OrderedCounter


class ChairsInContext(data.Dataset):

    def __init__(
            self, 
            data_dir, 
            data_size = None,
            vocab = None, 
            split = 'train', 
            context_condition = 'all',
            split_mode = 'easy', 
            image_size = 32, 
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = None,
        ):

        super().__init__()

        assert split_mode in ['easy', 'hard']
        assert context_condition in ['all', 'far', 'close']

        self.data_dir = data_dir
        self.data_size = data_size
        self.vocab = vocab
        self.split = split
        self.context_condition = context_condition
        self.split_mode = split_mode
        self.image_size = image_size
        self.train_frac = train_frac
        self.val_frac = val_frac
        if image_transform is None:
            self.image_transform = transforms.ToTensor()
        else:
            self.image_transform = image_transform

        csv_path = os.path.join(self.data_dir, 'chairs2k_group_data.csv')
        df = pd.read_csv(csv_path)
        df = df[df['correct'] == True]
        df = df[df['communication_role'] == 'speaker']
        
        if self.context_condition != 'all':
            df = df[df['context_condition'] == self.context_condition]
        
        df = df[['chair_a', 'chair_b', 'chair_c', 'target_chair', 'text']]
        df = df.dropna()

        data = np.asarray(df)
        data_names = self._get_chair_image_names()
        data = self._clean_data(data, data_names)

        if self.data_size is not None:
            data = data[:self.data_size]
    
        if self.split_mode == 'easy':
            # for each unique chair, divide all rows containing it into training and test sets
            data = self._process_easy(data)
        elif self.split_mode == 'hard':
            data = self._process_hard(data)
        else:
            raise Exception(f'split_mode {self.split_mode} not supported.')
    
        labels = self._process_labels(data)

        text = [d[-1] for d in data]
        
        if vocab is None:
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

        text_seq, text_len, max_len, text_raw = self._process_text(text)

        self.data = data
        self.labels = labels
        self.text_seq = text_seq
        self.text_len = text_len
        self.max_len = max_len
        self.text_raw = text_raw
        self.data_names = data_names
        self.text = text

    def _get_chair_image_names(self):
        image_paths = glob(os.path.join(self.data_dir, '*.png'))
        names = [os.path.basename(path) for path in image_paths]
        return names

    def _clean_data(self, data, names):
        cleaned = []
        for i in range(len(data)):
            chair_a, chair_b, chair_c, chair_target, _ = data[i]
            
            if chair_a + '.png' not in names:
                continue
            if chair_b + '.png' not in names:
                continue
            if chair_c + '.png' not in names:
                continue
            if chair_target + '.png' not in names:
                continue

            cleaned.append(data[i])

        cleaned = np.array(cleaned)
        return cleaned

    def _process_easy(self, data):
        # for each unique chair, divide all rows containing it into training and test sets
        target_names = data[:, 3]
        target_uniqs = np.unique(target_names)

        processed = []
        for target in target_uniqs:
            data_i = data[target_names == target]
            n_train = int(self.train_frac * len(data_i))
            n_val = int((self.train_frac + self.val_frac) * len(data_i))
            
            if self.split == 'train':
                processed.append(data_i[:n_train])
            elif self.split == 'val':
                processed.append(data_i[n_train:n_val])
            elif self.split == 'test':
                processed.append(data_i[n_val:])
            else:
                raise Exception(f'split {self.split} not supported.')
        
        processed = np.concatenate(processed, axis=0)
        
        return processed

    def _process_hard(self, data):
        # for all chairs, divide into train and test sets
        target_names = data[:, 3]
        target_uniqs = np.unique(target_names)

        n_train = len(self.train_frac * len(target_uniqs))
        n_val = int((self.train_frac + self.val_frac) * len(data))
        
        if self.split == 'train':
            splitter = np.in1d(target_names, target_uniqs[:n_train])
        elif self.split == 'val':
            splitter = np.in1d(target_names, target_uniqs[n_train:n_val])
        elif self.split == 'test':
            splitter = np.in1d(target_names, target_uniqs[n_val:])
        else:
            raise Exception(f'split {self.split} not supported.')        

        processed = data[splitter]

        return processed

    def _process_labels(self, data):
        labels = []
        for i in range(data):
            if data[i, 3] == data[i, 0]:
                labels.append(0)
            elif data[i, 3] == data[i, 1]:
                labels.append(1)
            elif data[i, 3] == data[i, 2]:
                labels.append(2)
            else:
                raise Exception('Bad label')
        
        labels = np.array(labels)
        return labels

    def build_vocab(self, texts):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        for text in texts:
            tokens = word_tokenize(text)
            w2c.update(tokens)

        for w, c in w2c.items():
            i2w[len(w2i)] = w
            w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        return vocab

    def _process_text(self, text):
        text_seq, text_len, raw_tokens = [], [], []

        max_len = 0
        for i in range(len(text)):
            _tokens = word_tokenize(text[i])
            tokens = [SOS_TOKEN] + _tokens + [EOS_TOKEN]
            length = len(tokens)
            max_len = max(max_len, length)

            text_seq.append(tokens)
            text_len.append(length)
            raw_tokens.append(_tokens)

        for i in range(len(text)):
            tokens = text_seq[i]
            length = text_len[i]
            tokens.extend([PAD_TOKEN] * (max_len - length))
            tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens]
            text_seq[i] = tokens

        text_seq = np.array(text_seq)
        text_len = np.array(text_len)

        return text_seq, text_len, max_len, raw_tokens

    def __len__(self):
        return len(self.data)
    
    def __gettext__(self, index):
        return self.text_raw[index]

    def __getitem__(self, index):
        chair_a, chair_b, chair_c, _, _ = self.data[index]
        label = self.labels[index]

        chair_a_pil = Image.open(os.path.join(self.data_dir, chair_a)).convert('RGB')
        chair_b_pil = Image.open(os.path.join(self.data_dir, chair_b)).convert('RGB')
        chair_c_pil = Image.open(os.path.join(self.data_dir, chair_c)).convert('RGB')

        chair_a_pt = self.image_transform(chair_a_pil)
        chair_b_pt = self.image_transform(chair_b_pil)
        chair_c_pt = self.image_transform(chair_c_pil)

        text_seq = torch.from_numpy(self.text_seq[index]).long()
        text_len = torch.from_numpy(self.text_len[index]).long()

        return index, chair_a_pt, chair_b_pt, chair_c_pt, text_seq, text_len, label
