import os
import numpy as np
import torch

from reference.agents import FeatureAgent


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
    
    roberta = torch.hub.load('pytorch/fairseq', 'roberta.large')
    roberta.eval()

    def extract(raw_text_list):
        features = []
        for raw_text in raw_text_list:
            raw_text = ' '.join(raw_text)
            tokens = roberta.encode(raw_text)
            # last layer features [1, 5, 1024]
            last_layer_features = roberta.extract_features(tokens)
            sent_features = last_layer_features.mean(1)  # take a mean over the words
            features.append(sent_features.detach().cpu())
        features = torch.cat(features, dim=0)
        return features

    train_text_embs = agent.extract_features(extract, modality='text', split='train')
    val_text_embs = agent.extract_features(extract, modality='text', split='val')
    test_text_embs = agent.extract_features(extract, modality='text', split='test')

    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/roberta/train.npy', train_text_embs.numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/roberta/val.npy', val_text_embs.numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/roberta/test.npy', test_text_embs.numpy())
