import os
import time
import nltk
import pickle
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from reference.agents import FeatureAgent

GLOVE_PATH = '/mnt/fs5/wumike/reference/glove/glove.840B.300d.txt'


def get_glove(word_dict):
    word_vec = {}
    with open(GLOVE_PATH, 'r') as f:
        for line in f:
            values = line.split()
            word = values[0]
            if word in word_dict:
                if len(values) > 301:
                    print(f'Invalid entry: {word}')
                    continue
                vector = np.asarray(values[1:], "float32")
                word_vec[word] = vector
    print('Found %s(/%s) words with GloVe vectors' % (len(word_vec), len(word_dict)))
    return word_vec


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--context-condition', type=str, default='all', 
                        choices=['all', 'far', 'close'])
    parser.add_argument('--split-mode', type=str, default='easy',
                        choices=['easy', 'hard'])
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    agent = FeatureAgent(
        args.dataset,
        args.data_dir,
        context_condition = args.context_condition,
        split_mode = args.split_mode,
        image_size = 64,
        override_vocab = None, 
        batch_size = args.batch_size,
        gpu_device = args.gpu_device, 
        cuda = args.cuda,
        seed = args.seed,
        image_transforms = None,
    )
    word_dict = agent.train_dataset.vocab['w2i']
    word_vec = get_glove(word_dict)

    out_dir = f"/mnt/fs5/wumike/reference/pretrain/{args.dataset}/glove"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    with open(f'{out_dir}/word_dict.pickle', 'wb') as fp:
        pickle.dump(word_vec, fp)

    def extract(raw_text_list):
        batch_embs = []
        for raw_text in raw_text_list:
            embeddings = []
            for token in raw_text:
                if token in word_vec:
                    embeddings.append(word_vec[token])
            if len(embeddings) == 0:
                embeddings = np.zeros(300)  # some filler value
            else:
                embeddings = np.stack(embeddings).mean(0)  # take mean
            embeddings = torch.from_numpy(embeddings).float()
            batch_embs.append(embeddings)
        batch_embs = torch.stack(batch_embs)
        return batch_embs

    train_text_embs = agent.extract_features(extract, modality='text', split='train')
    val_text_embs = agent.extract_features(extract, modality='text', split='val')
    test_text_embs = agent.extract_features(extract, modality='text', split='test')

    np.save(f'{out_dir}/train.npy', train_text_embs.numpy())
    np.save(f'{out_dir}/val.npy', val_text_embs.numpy())
    np.save(f'{out_dir}/test.npy', test_text_embs.numpy())
