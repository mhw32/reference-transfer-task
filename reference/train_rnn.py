import os
import json
import numpy as np

import os
from copy import deepcopy
from pprint import pprint
from reference.agents import TrainAgent
from reference.setup import _process_config
from reference.setup import load_json

CUR_DIR = os.path.dirname(__file__)


def run_model(config_dict):
    config = _process_config(config_dict)
    agent = TrainAgent(config)
    agent.run()
    agent.finalise()


def build_config(
        exp_base, 
        exp_name,
        dataset,
        data_dir,
        data_size = None,
        context_condition = 'all',
        split_mode = 'easy',
        sneak_peak = False,
        pretrain_image_embedding_dir = None,
        pretrain_text_embedding_dir = None,
        pretrain_image_dim = None,
        pretrain_text_dim = None,
        auto_schedule = True,
        gpu_device = 0,
        cuda = False,
        seed = 42,
    ):
    config_dict = {
        "exp_base": exp_base,
        "exp_name": exp_name,
        "cuda": cuda,
        "gpu_device": gpu_device,
        "seed": seed,
        "data_loader_workers": 8,
        "dataset": dataset,
        "data_dir": data_dir,
        "data": {
            "data_size": data_size,
            "image_size": 64,
            "context_condition": context_condition,
            "split_mode": split_mode
        },
        "train_image_from_scratch": pretrain_image_embedding_dir is None,
        "train_text_from_scratch": pretrain_text_embedding_dir is None,
        "pretrain_image_embedding_dir": pretrain_image_embedding_dir,
        "pretrain_text_embedding_dir": pretrain_text_embedding_dir,
        "model": {
            "n_bottleneck": 128,
            "image": {
                "n_pretrain_image": pretrain_image_dim,
                "n_image_channels": 3,
                "n_conv_filters": 64
            },
            "text": {
                "n_pretrain_text": pretrain_text_dim,
                "n_embedding": 64,
                "n_gru_hidden": 128,
                "gru_bidirectional": False,
                "n_gru_layers": 1,
                "sneak_peak": sneak_peak
            }
        },
        "optim": {
            "optimizer": "Adam",
            "val_freq": 1,
            "auto_schedule": False,
            "batch_size": 128, 
            "learning_rate": 0.0005,
            "momentum": 0.9,
            "weight_decay": 0,
            "patience": 10,
            "epochs": 100
        }
    }

    return config_dict 


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['chairs_in_context', 'colors_in_context', 'colorgrids_in_context'])
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    exp_base = '/mnt/fs5/wumike/reference/trained_models/12_22'
    data_dir = '/mnt/fs5/wumike/datasets'

    for data_size in [None, 0.5, 0.25, 0.1, 0.05, 0.01]:
        exp_name = f'{args.dataset}_vanilla+_vanilla+_size{data_size}_seed{args.seed}'
        
        config_dict = build_config(
            exp_base, 
            exp_name,
            args.dataset,
            data_dir,
            data_size = data_size,
            context_condition = 'all',
            split_mode = 'easy',
            sneak_peak = True,
            pretrain_image_embedding_dir = None,
            pretrain_text_embedding_dir = None,
            pretrain_image_dim = 128,
            pretrain_text_dim = 128,
            auto_schedule = True,
            gpu_device = args.gpu_device,
            cuda = args.cuda,
            seed = args.seed,
        )
        print('===================')
        print(exp_name)
        print('===================')
        run_model(config_dict)
