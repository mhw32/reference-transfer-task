import os
import time
import nltk
import numpy as np
from tqdm import tqdm
from nltk import word_tokenize
from torchvision import transforms
from torchvision.models import vgg19

import torch
import torch.nn as nn

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
    
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])

    agent = FeatureAgent(
        args.dataset,
        args.data_dir,
        context_condition = args.context_condition,
        split_mode = args.split_mode,
        image_size = None,
        override_vocab = None, 
        batch_size = args.batch_size,
        gpu_device = args.gpu_device, 
        cuda = args.cuda,
        seed = args.seed,
        image_transforms = image_transforms,
    )

    model = vgg19(pretrained=True).to(agent.device)
    model.classifier = model.classifier[0]
    model.eval()

    def extract(chair):
        with torch.no_grad():
            return model(chair)

    train_chair_a, train_chair_b, train_chair_c = agent.extract_features(
        extract, modality='image', split='train')

    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/train_chair_a.npy', train_chair_a.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/train_chair_b.npy', train_chair_b.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/train_chair_c.npy', train_chair_c.cpu().numpy())

    val_chair_a, val_chair_b, val_chair_c = agent.extract_features(
        extract, modality='image', split='val')

    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/val_chair_a.npy', val_chair_a.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/val_chair_b.npy', val_chair_b.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/val_chair_c.npy', val_chair_c.cpu().numpy())
    
    test_chair_a, test_chair_b, test_chair_c = agent.extract_features(
        extract, modality='image', split='test')
    
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/test_chair_a.npy', test_chair_a.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/test_chair_b.npy', test_chair_b.cpu().numpy())
    np.save(f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/vgg/test_chair_c.npy', test_chair_c.cpu().numpy())

