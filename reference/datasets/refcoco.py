import os
import sys
import math
import json
import copy
import pickle
import numpy as np
import pandas as pd
from PIL import Image
from glob import glob
from tqdm import tqdm
import skimage.io as io
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


class CocoInContext(data.Dataset):

    def __init__(
            self,
            data_dir, 
            data_size = None,
            image_size = 224,
            vocab = None, 
            split = 'train', 
            train_frac = 0.64,
            val_frac = 0.16,
            image_transform = None,
            min_token_occ = 2,
            max_sent_len = 33,
            random_seed = 42,
            **kwargs
        ):

        super().__init__()

        if data_size is not None:
            assert data_size > 0
            assert data_size <= 1

        self.data_dir = data_dir
        self.data_size = data_size
        self.image_size = image_size
        self.vocab = vocab
        self.split = split
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.min_token_occ = min_token_occ
        self.max_sent_len = max_sent_len
        self.random_seed = random_seed

        pickle_path = os.path.join(self.data_dir, f'{self.split}.pickle')
        with open(pickle_path, 'rb') as fp:
            data_pickle = pickle.load(fp)

        images = data_pickle['img_paths']
        masks = data_pickle['masks']
        texts = data_pickle['texts']

        assert len(images) == len(masks)
        assert len(images) == len(texts)
        size = len(images)

        if data_size is not None:
            assert split == 'train'
            rs = np.random.RandomState(self.random_seed)
            indices = np.arange(size)
            size = int(math.ceil(data_size * size))
            indices = rs.choice(indices, size=size)
            images = [images[idx] for idx in indices]
            masks = [masks[idx] for idx in indices]
            texts = [texts[idx] for idx in indices]

        if vocab is None:
            print('Building vocabulary')
            
            def flatten_texts(img_texts):
                all_texts = []
                for i in range(len(img_texts)):
                    texts = img_texts[i]
                    texts = [' '.join(text) for text in texts]
                    all_texts.extend(texts)
                return all_texts

            flat_texts = flatten_texts(texts)
            self.vocab = self.build_vocab(flat_texts)
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

        # right now, these are stored 
        #   [img_path, ...]
        #   [[bunch of masks], ...]
        #   [[bunch of texts], ...]
        # 
        # we want to get these into
        #   [img_path, ...]
        #   [[mask_a, mask_b, ...], ...]
        #   [[text_a, text_b, ...], ...]
        #   [label, ...]
        images, masks, texts, labels = self._unroll_data(
            images, 
            masks,
            texts,
            min_num_obj=3,  # ignore images with < 3 objects
        )

        # convert raw tokens -> vector of vocabulary indices
        text_seqs, text_lens, text_raws = self._process_text(texts)

        self.images = images
        self.masks = masks
        self.texts = texts
        self.labels = labels
        self.text_seqs = text_seqs
        self.text_lens = text_lens
        self.text_raws = text_raws
        self.size = len(images)

    def _unroll_data(self, images, masks, texts, min_num_obj=3):
        assert len(images) == len(masks)
        assert len(images) == len(texts)

        all_images = []
        all_masks  = []
        all_texts  = []
        all_labels = []

        for i in tqdm(range(len(images))):
            image, mask, text = images[i], masks[i], texts[i]
            assert len(mask) == len(text)
            
            if len(mask) < min_num_obj:
                continue

            # TODO: what to do about > 3 objects?

            for j in range(len(mask)):
                mask_j = copy.deepcopy(mask)
                text_j = text[j]
                label = j

                all_images.append(image)
                all_masks.append(mask_j)
                all_texts.append(text_j)
                all_labels.append(label)

        return all_images, all_masks, all_texts, all_labels

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
            tokens = text.lower().split(' ')
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
            sent_seqs, sent_lens, sent_tokens = [], [], []

            sents = text[i]
            for sent in sents:
                _tokens = [tok.lower() for tok in sent]
                tokens = [SOS_TOKEN] + _tokens[:self.max_sent_len] + [EOS_TOKEN]
                length = len(tokens)
                tokens.extend([PAD_TOKEN] * (self.max_sent_len + 2 - length))
                tokens = [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens]

                sent_seqs.append(tokens)
                sent_lens.append(length)
                sent_tokens.append(_tokens)
            
            text_seq.append(sent_seqs)
            text_len.append(sent_lens)
            raw_tokens.append(sent_tokens)

        return text_seq, text_len, raw_tokens

    def __len__(self):
        return self.size

    def __gettext__(self, index):
        return ' '.join(self.text_raws[index])

    def __getitem__(self, index):
        image = self.images[index]
        masks = self.masks[index]
        label = self.labels[index]
        text_seq = self.text_seqs[index]
        text_len = self.text_lens[index]

        image = Image.open(os.path.join(self.data_dir, image))

        if self.image_transform is None:
            image_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
            ])
            image = image_transform(image)
        else:
            image = self.image_transform(image)

        text_seq = torch.from_numpy(np.array(text_seq)).long()

        return index, image, mask, text_seq, text_len, label
