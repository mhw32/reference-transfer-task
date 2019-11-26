import os
import time
import nltk
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize

import torch
import torch.nn as nn

from reference.agents import FeatureAgent

W2V_PATH = '/mnt/fs5/wumike/reference/fastText/crawl-300d-2M.vec'


def get_w2v(word_dict):
    word_vec = {}
    with open(W2V_PATH, encoding='utf-8') as f:
        for line in f:
            word, vec = line.split(' ', 1)
            if word in word_dict:
                word_vec[word] = np.fromstring(vec, sep=' ')
    print('Found %s(/%s) words with w2v vectors' % (len(word_vec), len(word_dict)))
    return word_vec


if __name__ == "__main__":
    agent = FeatureAgent()
    word_dict = agent.train_dataset.vocab['w2i']
    word_vec = get_w2v(word_dict)

    def extract(raw_text_list):
        batch_embs = []
        for raw_text in raw_text_list:
            tokens = word_tokenize(raw_text)
            embeddings = [word_vec[token] for token in tokens]
            embeddings = np.stack(embeddings).mean(0)  # take mean
            embeddings = torch.from_numpy(embeddings).float()
            batch_embs.append(embeddings)
        batch_embs = torch.stack(batch_embs)
        return batch_embs

    train_text_embs = agent.extract_features(extract, modality='text', split='train')
    val_text_embs = agent.extract_features(extract, modality='text', split='val')
    test_text_embs = agent.extract_features(extract, modality='text', split='test')

    np.save('/mnt/fs5/wumike/reference/pretrain/word2vec/train.npy', train_text_embs.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/word2vec/val.npy', val_text_embs.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/word2vec/test.npy', test_text_embs.numpy())
