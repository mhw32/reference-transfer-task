import os
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from reference.agents import FeatureAgent, MaskedFeatureAgent

CUR_DIR = os.path.dirname(__file__)
LOCALAGG_DIR = os.path.realpath(os.path.join(CUR_DIR, '../localagg'))
MODEL_DIR = "/mnt/fs5/wumike/localagg/trained_models/7_12/experiments/imagenet_la/2019-08-17--11_00_06"

sys.path.append(LOCALAGG_DIR)
from src.agents.agents import *
from src.utils.setup import process_config


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

    OUT_DIR = f"/mnt/fs5/wumike/reference/pretrain/{args.dataset}/la_imagenet"
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)

    config_path = os.path.join(MODEL_DIR, 'config.json')
    checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
    assert os.path.isfile(os.path.join(checkpoint_dir, 'model_best.pth.tar'))

    config = process_config(config_path, override_dotmap={'gpu_device': args.gpu_device})
    AgentClass = globals()[config.agent]
    localagg = AgentClass(config)
    localagg.load_checkpoint(
        'model_best.pth.tar', 
        checkpoint_dir = checkpoint_dir, 
        load_memory_bank = True, 
        load_model = True,
    )
    localagg._set_models_to_eval()
    gpu_device = localagg.config.gpu_device[0]
    resnet = copy.deepcopy(localagg.model)
    resnet.load_state_dict(localagg.model.state_dict())
    # resnet = nn.Sequential(*list(resnet.children())[:-2])
    resnet = resnet.eval()
    for param in resnet.parameters():
        param.requires_grad = False

    image_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean = [0.485, 0.456, 0.406],
            std = [0.229, 0.224, 0.225],
        ),
    ])

    if args.dataset in ['refclef', 'refcoco', 'refcoco+']:
        FeatureAgentClass = MaskedFeatureAgent
    else:
        FeatureAgentClass = FeatureAgent

    agent = FeatureAgentClass(
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
    
    def extract(chair):
        with torch.no_grad():
            batch_size = chair.size(0)
            chair = chair.to(localagg.device)
            return resnet(chair).view(batch_size, -1).cpu()
    
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
