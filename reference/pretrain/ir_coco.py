import os
import sys
import copy
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms

from reference.agents import FeatureAgent

GPU_DEVICE = 0
CUR_DIR = os.path.dirname(__file__)
LOCALAGG_DIR = os.path.realpath(os.path.join(CUR_DIR, '../localagg'))
MODEL_DIR = "/mnt/fs5/wumike/localagg/trained_models/10_11/experiments/coco_composite_lrdrop2/2019-10-27--10_37_43"
OUT_IMG_DIR = "/mnt/fs5/wumike/reference/pretrain/ir_coco_image"
OUT_TXT_DIR = "/mnt/fs5/wumike/reference/pretrain/ir_coco_text"

sys.path.append(LOCALAGG_DIR)
from src.agents.agents import *
from src.utils.setup import process_config

config_path = os.path.join(MODEL_DIR, 'config.json')
checkpoint_dir = os.path.join(MODEL_DIR, 'checkpoints')
assert os.path.isfile(os.path.join(checkpoint_dir, 'model_best.pth.tar'))

config = process_config(config_path, override_dotmap={'gpu_device': GPU_DEVICE})
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
resnet = copy.deepcopy(localagg.img_model)
resnet.load_state_dict(localagg.img_model.state_dict())
resnet = nn.Sequential(*list(resnet.children())[:-2])
resnet = resnet.eval()
for param in resnet.parameters():
    param.requires_grad = False

rnn = copy.deepcopy(localagg.txt_model)
rnn.load_state_dict(localagg.txt_model.state_dict())
rnn = rnn.eval()
for param in rnn.parameters():
    param.requires_grad = False

override_vocab = copy.deepcopy(localagg.train_dataset.vocab)


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

    agent = FeatureAgent(
        override_vocab = override_vocab,
        image_transforms = image_transforms,
        gpu_device = gpu_device,
    )
    
    def extract_img(chair):
        with torch.no_grad():
            batch_size = chair.size(0)
            chair = chair.to(localagg.device)
            return resnet(chair).view(batch_size, -1).cpu()
   
    train_chair_a, train_chair_b, train_chair_c = agent.extract_features(
        extract_img, modality='image', split='train')

    np.save(f'{OUT_IMG_DIR}/train_chair_a.npy', train_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/train_chair_b.npy', train_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/train_chair_c.npy', train_chair_c.cpu().numpy())

    val_chair_a, val_chair_b, val_chair_c = agent.extract_features(
        extract_img, modality='image', split='val')

    np.save(f'{OUT_IMG_DIR}/val_chair_a.npy', val_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/val_chair_b.npy', val_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/val_chair_c.npy', val_chair_c.cpu().numpy())
    
    test_chair_a, test_chair_b, test_chair_c = agent.extract_features(
        extract_img, modality='image', split='test')
    
    np.save(f'{OUT_IMG_DIR}/test_chair_a.npy', test_chair_a.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/test_chair_b.npy', test_chair_b.cpu().numpy())
    np.save(f'{OUT_IMG_DIR}/test_chair_c.npy', test_chair_c.cpu().numpy())

    def extract_txt(text_seq, text_len):
        with torch.no_grad():
            text_seq = text_seq.to(localagg.device)
            text_len = text_len.to(localagg.device)
            return rnn(text_seq, text_len).cpu()
    
    train_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='train')
    val_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='val')
    test_text_embs = agent.extract_features(extract_txt, modality='encoded_text', split='test')

    np.save(f'{OUT_TXT_DIR}/train.npy', train_text_embs.numpy())
    np.save(f'{OUT_TXT_DIR}/val.npy', val_text_embs.numpy())
    np.save(f'{OUT_TXT_DIR}/test.npy', test_text_embs.numpy())
