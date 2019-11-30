import os
import sys
import copy
import numpy as np
from tqdm import tqdm
from dotmap import DotMap
import torch
import torch.nn as nn
from torchvision import transforms

from reference.agents import FeatureAgent

GPU_DEVICE = 1
CUR_DIR = os.path.dirname(__file__)
MVAE_DIR = os.path.realpath(os.path.join(CUR_DIR, '../mvae'))
MODEL_DIR = "/mnt/fs5/wumike/hybrid/trained_models/8_22/longtests/experiments/TrainAgent_coco_vae_seed1337/2019-08-22--11_31_53"
OUT_DIR = "/mnt/fs5/wumike/reference/pretrain/vae"

sys.path.append(MVAE_DIR)
from src.agents.agents import *
from src.utils.utils import load_json

config_path = os.path.join(MODEL_DIR, 'config.json')
checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
assert os.path.isfile(os.path.join(checkpoint_dir, 'model_best.pth.tar'))

config = load_json(config_path)
config['gpu_device'] = GPU_DEVICE
config = DotMap(config)

AgentClass = globals()[config.agent]
mvae = AgentClass(config)
mvae.load_checkpoint('model_best.pth.tar')
mvae._set_models_to_eval()
gpu_device = mvae.config.gpu_device


if __name__ == "__main__":
    image_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])

    agent = FeatureAgent(
        image_transforms = image_transforms,
        gpu_device = gpu_device,
    )

    def extract(chair):
        with torch.no_grad():
            chair = chair.to(mvae.device)
            z_mu, _ = mvae.model_dicts[0]['vae_inf'](chair)
            return z_mu.cpu()
    
    train_chair_a, train_chair_b, train_chair_c = agent.extract_features(
        extract, modality='image', split='train')

    np.save(f'{OUT_DIR}/train_chair_a.npy', train_chair_a.cpu().numpy())
    np.save(f'{OUT_DIR}/train_chair_b.npy', train_chair_b.cpu().numpy())
    np.save(f'{OUT_DIR}/train_chair_c.npy', train_chair_c.cpu().numpy())

    val_chair_a, val_chair_b, val_chair_c = agent.extract_features(
        extract, modality='image', split='val')

    np.save(f'{OUT_DIR}/val_chair_a.npy', val_chair_a.cpu().numpy())
    np.save(f'{OUT_DIR}/val_chair_b.npy', val_chair_b.cpu().numpy())
    np.save(f'{OUT_DIR}/val_chair_c.npy', val_chair_c.cpu().numpy())
    
    test_chair_a, test_chair_b, test_chair_c = agent.extract_features(
        extract, modality='image', split='test')
    
    np.save(f'{OUT_DIR}/test_chair_a.npy', test_chair_a.cpu().numpy())
    np.save(f'{OUT_DIR}/test_chair_b.npy', test_chair_b.cpu().numpy())
    np.save(f'{OUT_DIR}/test_chair_c.npy', test_chair_c.cpu().numpy())
