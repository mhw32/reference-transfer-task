import os
import sys
import math
import json
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

sys.path.append(os.path.dirname(__file__))
from refer import REFER


class RefCOCOInContext(data.Dataset):

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

        data_name = os.path.basename(data_dir)
        data_dir = os.path.dirname(data_dir)
        
        assert split in ['train', 'val', 'test']
        assert data_name in ['refclef', 'refcoco', 'refcoco+']
        
        self.refer = REFER(data_dir, data_name, 'google')
        self.ref_ids = self.refer.getRefIds(split=self.split)

        if data_size is not None:
            rs = np.random.RandomState(self.random_seed)
            n_train_total = len(self.ref_ids)
            indices = np.arange(n_train_total)
            n_train_total = int(math.ceil(data_size * n_train_total))
            indices = rs.choice(indices, size=n_train_total)
            self.ref_ids = [self.ref_ids[idx] for idx in indices]

        self.img_ids = self.refer.getImgIds(ref_ids=self.ref_ids)
        self.img_refs = [self.refer.imgToRefs[img_id] for img_id in self.img_ids]
        self.img_masks = self.get_masks(self.img_refs)
        self.img_texts = self.get_text(self.img_refs)

        self.flat_img_ids = []
        self.flat_img_ctxs = []
        self.flat_img_refs = []
        self.flat_img_masks = []
        self.flat_img_texts = []

        for i in range(len(self.img_ids)):
            img_refs = self.img_refs[i]
            img_masks = self.img_masks[i]
            img_texts = self.img_texts[i]
            img_id = self.img_ids[i]

            assert len(img_refs) == len(img_masks)
            assert len(img_refs) == len(img_texts)

            img_ctxs = []

            self.flat_img_refs.extend(img_refs)
            self.flat_img_masks.extend(img_masks)
            self.flat_img_texts.extend(img_texts)
            self.flat_img_ids.extend([img_id for _ in range(len(img_refs))])

        if vocab is None:
            print('Building vocabulary')
            self.vocab = self.build_vocab(self.flat_img_texts)
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

        flat_img_text_seqs, flat_img_text_lens, flat_img_text_raws = \
            self._process_text(flat_img_texts)

        self.flat_img_text_seqs = flat_img_text_seqs
        self.flat_img_text_lens = flat_img_text_lens
        self.flat_img_text_raws = flat_img_text_raws

    def get_masks(self, imgToRefs):
        imgToMasks = []
        for refs in imgToRefs:
            masks = [self.refer.getMask(ref) for ref in refs]
            imgToMasks.append(masks)
        return imgToMasks

    def get_text(self, imgToRefs):
        ref_sents = []
        flat_sents = []
        for refs in imgToRefs:
            sents = []
            for ref in refs:
                # just take the first sentence
                sent = ref['sentences'][0]
                sent = ' '.join(sent['tokens'])
                sents.append(sent)
                flat_sents.append(sent)
            ref_sents.append(sents)
        return ref_sents

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
            sents = text[i]
            sent_seqs = []
            sent_lens = []
            sent_tokens = []

            for sent in sents:
                _tokens = sent.lower().split(' ')
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
        return len(self.flat_img_ids)

    def __gettext__(self, index):
        return self.flat_img_text_raws[index]

    def __getitem__(self, index):
        img_id = self.img_ids[index]
        image = self.refer.Imgs[img_id]
        image = Image.open(os.path.join(self.data_dir, image['file_name']))
        
        if self.image_transform is None:
            image = transforms.ToTensor()(image)
        else:
            image = self.image_transform(image)

        mask = self.flat_img_masks[img_id]
        text_seqs = self.flat_img_text_seqs[img_id]
        text_lens = self.flat_img_text_lens[img_id]
        text_raws = self.flat_img_text_raws[img_id]

        text_seq = torch.from_numpy(np.array(text_seqs)).long()

        return index, image, mask, text_seq, text_len
