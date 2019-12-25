import os
import math
import json
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
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
from reference.datasets.colors import clean_tokens, hsl2rgb


class ColorgridsInContext(data.Dataset):

    def __init__(
            self,
            data_dir, 
            data_size = None,
            image_size = 64,
            vocab = None, 
            split = 'train', 
            context_condition = 'all',
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = None,
            min_token_occ = 2,
            max_sent_len = 60,
            random_seed = 42,
            **kwargs
        ):

        super().__init__()
        assert context_condition in ['all', 'far', 'close', 'split']

        if image_size is None:
            image_size = 64

        if data_size is not None:
            assert data_size > 0
            assert data_size <= 1

        self.data_dir = data_dir
        self.cache_dir = os.path.join(self.data_dir, 'cache')
        self.data_size = data_size
        self.image_size = image_size
        self.vocab = vocab
        self.split = split
        self.context_condition = context_condition
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.min_token_occ = min_token_occ
        self.max_sent_len = max_sent_len
        self.random_seed = random_seed
        self.subset_indices = None

        if image_transform is None:
            self.image_transform = transforms.Compose([
                transforms.Resize(self.image_size),
                transforms.CenterCrop(self.image_size),
                transforms.ToTensor(),
            ])
        else:
            self.image_transform = image_transform

        if not os.path.isdir(self.cache_dir):
            os.makedirs(self.cache_dir)

        cache_clean_data = os.path.join(
            self.cache_dir,
            f'clean_data_{self.context_condition}.pickle',
        )

        if not os.path.isfile(cache_clean_data):
            raw_data = self._load_data(self.data_dir)
            texts, images1, images2, images3, labels = self._prune_data(raw_data, self.context_condition)
            with open(cache_clean_data, 'wb') as fp:
                pickle.dump({
                    'texts': texts,
                    'images1': images1,
                    'images2': images2,
                    'images3': images3,
                    'labels': labels,
                }, fp)
        else:
            print('Loading cleaned data from pickle.')
            with open(cache_clean_data, 'rb') as fp:
                cache = pickle.load(fp)
                texts = cache['texts']
                images1 = cache['images1']
                images2 = cache['images2']
                images3 = cache['images3']
                labels = cache['labels']

        data = list(zip(texts, images1, images2, images3, labels))
        data = self._process_splits(data)

        if data_size is not None:
            rs = np.random.RandomState(self.random_seed)
            n_train_total = len(data)
            indices = np.arange(n_train_total)
            n_train_total = int(math.ceil(data_size * n_train_total))
            indices = rs.choice(indices, size=n_train_total)
            data = [data[index] for index in indices]

            self.subset_indices = indices

        text = [d[0] for d in data]

        if vocab is None:
            print('Building vocabulary')
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

        text_seq, text_len, text_raw = self._process_text(text)

        self.data = data
        self.text_seq = text_seq
        self.text_len = text_len
        self.text_raw = text_raw
        self.text = text

    def _load_data(self, data_dir):
        data = []
        for line in open(os.path.join(data_dir, 'processed1.json'), 'r'):
            data.append(json.loads(line))
        for line in open(os.path.join(data_dir, 'processed2.json'), 'r'):
            data.append(json.loads(line))
        return data

    def _prune_data(self, data, context_condition):
        # only save utterances by the speaker 
        # only take rows where listener got the answer correct
        # parse by context_condition

        text_all, image1_all, image2_all, image3_all, labels_all = [], [], [], [], []

        print('Pruning data.')
        for records in data:
            records = records['records']
            for rounds in records:

                utterances = []
                image_event = None
                action_event = None
                            
                for event in rounds['events']:
                    if event['eventType'] == 'utterance':
                        # only take things that the speaker said
                        if event['sender'] != 'speaker':
                            continue
                        utterances.append(event['contents'])

                    elif event['eventType'] == 'action':
                        assert action_event is None
                        action_event = event

                    elif event['eventType'] == 'state':
                        assert image_event is None
                        image_event = event

                listener_choice = action_event['action']['lClicked']
                correct_choice = image_event['state']['target']
                
                if listener_choice != correct_choice:
                    continue

                condition = image_event['state']['condition']['name'].lower()
                if context_condition != 'all':  # take all conditions
                    if condition != context_condition:
                        continue
                
                text = ' '.join(utterances) 
                images = image_event['state']['objs']

                def construct_image(image):
                    shapes = image['shapes']
                    shapes = np.array([shape['color'] for shape in shapes])
                    shapes = shapes.reshape(3, 3, 3)
                    return shapes

                images = [construct_image(image) for image in images]
                assert len(images) == 3

                text_all.append(text)
                image1_all.append(images[0])
                image2_all.append(images[1])
                image3_all.append(images[2])
                labels_all.append(correct_choice)

        return text_all, image1_all, image2_all, image3_all, labels_all

    def _process_splits(self, data):
        rs = np.random.RandomState(self.random_seed)
        rs.shuffle(data)  # important

        n_train = int(self.train_frac * len(data))
        n_val = int((self.train_frac + self.val_frac) * len(data))

        if self.split == 'train':
            data = data[:n_train]
        elif self.split == 'val':
            data = data[n_train:n_val]
        elif self.split == 'test':
            data = data[n_val:]
        else:
            raise Exception(f'split {self.split} not supported.')
        
        return data

    def build_vocab(self, texts):
        w2i = dict()
        i2w = dict()
        w2c = OrderedCounter()
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        pbar = tqdm(total=len(texts))
        for text in texts:
            tokens = word_tokenize(text.lower())
            tokens = clean_tokens(tokens)
            w2c.update(tokens)
            pbar.update()
        pbar.close()

        for w, c in w2c.items():
            if c >= self.min_token_occ:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)
        vocab = dict(w2i=w2i, i2w=i2w)

        return vocab

    def _process_text(self, text):
        text_seq, text_len, raw_tokens = [], [], []

        for i in range(len(text)):
            _tokens = word_tokenize(text[i].lower())
            _tokens = clean_tokens(_tokens)
            
            tokens = [SOS_TOKEN] + _tokens[:self.max_sent_len] + [EOS_TOKEN]
            length = len(tokens)
            tokens.extend([PAD_TOKEN] * (self.max_sent_len + 2 - length))
            tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens]
            
            text_seq.append(tokens)
            text_len.append(length)
            raw_tokens.append(_tokens)

        text_seq = np.array(text_seq)
        text_len = np.array(text_len)

        return text_seq, text_len, raw_tokens

    def _make_image_from_coords(self, coords):
        p11, p12, p13 = coords[0]
        p21, p22, p23 = coords[1]
        p31, p32, p33 = coords[2]

        # convert hsl to rgb
        p11 = np.array(list(hsl2rgb(p11[0], p11[1] / 100., p11[2] / 100.)))
        p12 = np.array(list(hsl2rgb(p12[0], p12[1] / 100., p12[2] / 100.)))
        p13 = np.array(list(hsl2rgb(p13[0], p13[1] / 100., p13[2] / 100.)))
        p21 = np.array(list(hsl2rgb(p21[0], p21[1] / 100., p21[2] / 100.)))
        p22 = np.array(list(hsl2rgb(p22[0], p22[1] / 100., p22[2] / 100.)))
        p23 = np.array(list(hsl2rgb(p23[0], p23[1] / 100., p23[2] / 100.)))
        p31 = np.array(list(hsl2rgb(p31[0], p31[1] / 100., p31[2] / 100.)))
        p32 = np.array(list(hsl2rgb(p32[0], p32[1] / 100., p32[2] / 100.)))
        p33 = np.array(list(hsl2rgb(p33[0], p33[1] / 100., p33[2] / 100.)))

        # p11 = p11[np.newaxis, np.newaxis, :]
        # p12 = p12[np.newaxis, np.newaxis, :]
        # p13 = p13[np.newaxis, np.newaxis, :]
        # p21 = p21[np.newaxis, np.newaxis, :]
        # p22 = p22[np.newaxis, np.newaxis, :]
        # p23 = p23[np.newaxis, np.newaxis, :]
        # p31 = p31[np.newaxis, np.newaxis, :]
        # p32 = p32[np.newaxis, np.newaxis, :]
        # p33 = p33[np.newaxis, np.newaxis, :]

        size1 = self.image_size // 3
        size3 = self.image_size // 3
        size2 = self.image_size - size1 - size3

        p11 = p11[np.newaxis, np.newaxis, :]
        p11 = p11.repeat(size1, 0).repeat(size1, 1)
        p12 = p12[np.newaxis, np.newaxis, :]
        p12 = p12.repeat(size1, 0).repeat(size2, 1)
        p13 = p13[np.newaxis, np.newaxis, :]
        p13 = p13.repeat(size1, 0).repeat(size3, 1)

        p21 = p21[np.newaxis, np.newaxis, :]
        p21 = p21.repeat(size2, 0).repeat(size1, 1)
        p22 = p22[np.newaxis, np.newaxis, :]
        p22 = p22.repeat(size2, 0).repeat(size2, 1)
        p23 = p23[np.newaxis, np.newaxis, :]
        p23 = p23.repeat(size2, 0).repeat(size3, 1)

        p31 = p31[np.newaxis, np.newaxis, :]
        p31 = p31.repeat(size3, 0).repeat(size1, 1)
        p32 = p32[np.newaxis, np.newaxis, :]
        p32 = p32.repeat(size3, 0).repeat(size2, 1)
        p33 = p33[np.newaxis, np.newaxis, :]
        p33 = p33.repeat(size3, 0).repeat(size3, 1)

        r1 = np.concatenate((p11, p12, p13), axis=1)
        r2 = np.concatenate((p21, p22, p23), axis=1)
        r3 = np.concatenate((p31, p32, p33), axis=1)
        image = np.concatenate((r1, r2, r3), axis=0)

        return Image.fromarray(image.astype('uint8'))

    def __len__(self):
        return len(self.data)
    
    def __gettext__(self, index):
        return self.text_raw[index]

    def __getitem__(self, index):
        raw_text, _image1, _image2, _image3, label = self.data[index]

        image1 = self._make_image_from_coords(_image1)
        image2 = self._make_image_from_coords(_image2)
        image3 = self._make_image_from_coords(_image3)

        image1 = self.image_transform(image1)
        image2 = self.image_transform(image2)
        image3 = self.image_transform(image3)

        text_seq = torch.from_numpy(self.text_seq[index]).long()
        text_len = self.text_len[index]

        return index, image1, image2, image3, text_seq, text_len, label


if __name__ == "__main__":
    dataset = ColorgridsInContext('/mnt/fs5/wumike/datasets/colorgrids_in_context')
    print(dataset.__getitem__(25)[-1])
    print(dataset.__gettext__(25))
