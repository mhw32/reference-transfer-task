import os
import sys
import time
import nltk
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn

from reference.agents import FeatureAgent
from reference.setup import process_config

CUR_DIR = os.path.dirname(__file__)
SKIPTHOUGHT_DIR = os.path.realpath(
    os.path.join(CUR_DIR, '../skip-thoughts.torch/pytorch'))
sys.path.append(SKIPTHOUGHT_DIR)
SKIPTHOUGHT_PATH = '/mnt/fs5/wumike/reference/skipthought/data/skip-thoughts'

from skipthoughts import BiSkip


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

    vocab_list = list(agent.train_dataset.vocab['w2i'].keys())
    biskip = BiSkip(SKIPTHOUGHT_PATH, vocab_list)

    def extract(text_seq, text_len):
        with torch.no_grad():
            lengths = text_len.cpu().numpy().tolist()
            output = biskip(text_seq.cpu().long(), lengths=lengths)
        return output

    train_text_embs = agent.extract_features(extract, modality='encoded_text', split='train')
    val_text_embs = agent.extract_features(extract, modality='encoded_text', split='val')
    test_text_embs = agent.extract_features(extract, modality='encoded_text', split='test')

    OUT_DIR = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/skipthought'
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    np.save(f'{OUT_DIR}/train.npy', train_text_embs.numpy())
    np.save(f'{OUT_DIR}/val.npy', val_text_embs.numpy())
    np.save(f'{OUT_DIR}/test.npy', test_text_embs.numpy())
