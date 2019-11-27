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
    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])

    agent = FeatureAgent(image_transforms = image_transforms)

    model = vgg19(pretrained=True).to(agent.device)
    model.classifier = model.classifier[0]
    model.eval()

    def extract(chair):
        with torch.no_grad():
            return model(chair)

    train_chair_a, train_chair_b, train_chair_c = agent.extract_features(
        extract, modality='image', split='train')

    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/train_chair_a.npy', train_chair_a.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/train_chair_b.npy', train_chair_b.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/train_chair_c.npy', train_chair_c.numpy())

    val_chair_a, val_chair_b, val_chair_c = agent.extract_features(
        extract, modality='image', split='val')

    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/val_chair_a.npy', val_chair_a.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/val_chair_b.npy', val_chair_b.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/val_chair_c.npy', val_chair_c.numpy())
    
    test_chair_a, test_chair_b, test_chair_c = agent.extract_features(
        extract, modality='image', split='test')
    
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/test_chair_a.npy', test_chair_a.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/test_chair_b.npy', test_chair_b.numpy())
    np.save('/mnt/fs5/wumike/reference/pretrain/vgg/test_chair_c.npy', test_chair_c.numpy())

