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

GPU_DEVICE = 2
CUR_DIR = os.path.dirname(__file__)
MVAE_DIR = os.path.realpath(os.path.join(CUR_DIR, '../mvae'))
MODEL_DIR = "/mnt/fs5/wumike/hybrid/trained_models/8_22/longtests/experiments/TrainAgent_coco_vaegan_seed1337/2019-09-05--23_14_26"
OUT_IMG_DIR = "/mnt/fs5/wumike/reference/pretrain/vaegan_image"
OUT_TXT_DIR = "/mnt/fs5/wumike/reference/pretrain/vaegan_text"

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
gpu_device = mvae.config.gpu_device[0]

override_vocab = copy.deepcopy(mvae.train_dataset.vocab)


if __name__ == "__main__":
    image_transforms = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    agent = FeatureAgent(
        override_vocab = override_vocab,
        image_transforms = image_transforms,
        gpu_device = gpu_device,
    )

    def extract_img(chair):
        with torch.no_grad():
            chair = chair.to(mvae.device)
            z_mu, _ = mvae.model_dicts[0]['gan_inf'](chair)
            return z_mu.cpu()

    train_chair_a, train_chair_b, train_chair_c = agent.extract_features(
        extract_img, modality='image', split='train')

    np.save(f'{OUT_IMG_DIR}/train_chair_a.npy', train_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/train_chair_b.npy', train_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/train_chair_c.npy', train_chair_c.cpu().numpy())

    val_chair_a, val_chair_b, val_chair_c = agent.extract_features(
        extract_img, modality = 'image', split = 'val')

    np.save(f'{OUT_IMG_DIR}/val_chair_a.npy', val_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/val_chair_b.npy', val_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/val_chair_c.npy', val_chair_c.cpu().numpy())
    
    test_chair_a, test_chair_b, test_chair_c = agent.extract_features(
        extract_img, modality = 'image', split = 'test')
    
    np.save(f'{OUT_IMG_DIR}/test_chair_a.npy', test_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/test_chair_b.npy', test_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/test_chair_c.npy', test_chair_c.cpu().numpy())

    def extract_txt(text_seq, text_len):
        with torch.no_grad():
            text_seq = text_seq.to(mvae.device)
            text_len = text_len.to(mvae.device)
            z_mu, _ = mvae.model_dicts[0]['vae_inf'](text_seq, text_len)
            return z_mu.cpu()

    train_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='train')
    val_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='val')
    test_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='test')

    np.save(f'{OUT_TXT_DIR}/train.npy', train_text_embs.numpy())
    np.save(f'{OUT_TXT_DIR}/val.npy', val_text_embs.numpy())
    np.save(f'{OUT_TXT_DIR}/test.npy', test_text_embs.numpy())
