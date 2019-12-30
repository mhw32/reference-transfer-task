"""
Master script to sequentially train many many models. We purposefully
train all models with 100% supervision first, then start to reduce the 
amount of supervision. We train everytihng sequentially so this is quite
an expensive process. We exhaustively train every combination of models.
We assume all pretrained  embeddings have already been collected.
"""

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

LANGUAGE_MODELS = [
    'vanilla',
    'glove',
    'word2vec',
    'skipthought',
    'infersent',
    'huggingface/bert',
    'huggingface/gpt_openai',
    'huggingface/gpt_2',
    'huggingface/ctrl',
    'huggingface/transforxl',
    'huggingface/xlnet',
    'huggingface/xlm',
    'huggingface/distilbert',
    'huggingface/roberta',
]


IMAGE_MODELS = [
    'vae',
]

MULTIMODAL_MODELS = [
    'ir_coco',
    'vaevae',
    'vaegan',
]

PRETRAIN_IMAGE_DIM_DICT = {
    'vae': 256,
}

PRETRAIN_TEXT_DIM_DICT = {
    'vanilla': 128,
    'glove': 300,
    'word2vec': 300,
    'skipthought': 2400,
    'infersent': 4096,
    'huggingface/bert': 768,
    'huggingface/gpt_openai': 768,
    'huggingface/gpt_2': 768,
    'huggingface/ctrl': 1280,
    'huggingface/transforxl': 1024,
    'huggingface/xlnet': 768,
    'huggingface/xlm': 1024,
    'huggingface/distilbert': 768,
    'huggingface/roberta': 768,
}

PRETRAIN_MULTIMODAL_DIM_DICT = {
    'ir_coco': (128, 128),
    'vaevae': (256, 256),
    'vaegan': (256, 256),
}


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
    parser.add_argument('--data-size', default=None)
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--image-only', action='store_true', default=False)
    parser.add_argument('--text-only', action='store_true', default=False)
    parser.add_argument('--multimodal-only', action='store_true', default=False)
    args = parser.parse_args()

    exp_base = '/mnt/fs5/wumike/reference/trained_models/12_22'
    data_dir = '/mnt/fs5/wumike/datasets'

    if args.data_size is not None:
        args.data_size = float(args.data_size)

    if args.text_only or args.multimodal_only:
        image_models = ['vanilla']
    else:
        image_models = IMAGE_MODELS
        # image_models = image_models[1:]

    if args.image_only or args.multimodal_only:
        language_models = ['vanilla']
    else:
        language_models = LANGUAGE_MODELS

    if args.image_only or args.text_only:
        multimodal_models = []
    else:
        multimodal_models = MULTIMODAL_MODELS

    for image_model in image_models:
        for text_model in language_models:
            exp_name = f'{args.dataset}_{image_model}_{text_model}_size{args.data_size}_seed{args.seed}'
           
            if image_model == 'vanilla':
                pretrain_image_embedding_dir = None
            else:
                pretrain_image_embedding_dir = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/{image_model}'
            
            if text_model == 'vanilla':
                pretrain_text_embedding_dir = None
            else:
                pretrain_text_embedding_dir = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/{text_model}'
               
            config_dict = build_config(
                exp_base, 
                exp_name,
                args.dataset,
                data_dir,
                data_size = args.data_size,
                context_condition = 'all',
                split_mode = 'easy',
                pretrain_image_embedding_dir = pretrain_image_embedding_dir,
                pretrain_text_embedding_dir = pretrain_text_embedding_dir,
                pretrain_image_dim = PRETRAIN_IMAGE_DIM_DICT[image_model],
                pretrain_text_dim = PRETRAIN_TEXT_DIM_DICT[text_model],
                auto_schedule = True,
                gpu_device = args.gpu_device,
                cuda = args.cuda,
                seed = args.seed,
            )
            print('===================')
            print(exp_name)
            print('===================')
            run_model(config_dict)

    for multimodal_model in multimodal_models:
        exp_name = f'{multimodal_model}'

        pretrain_image_embedding_dir = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/{multimodal_model}_image'
        pretrain_text_embedding_dir = f'/mnt/fs5/wumike/reference/pretrain/{args.dataset}/{multimodal_model}_text'
            
        config_dict = build_config(
            exp_base, 
            exp_name,
            args.dataset,
            data_dir,
            data_size = args.data_size,
            context_condition = 'all',
            split_mode = 'easy',
            pretrain_image_embedding_dir = pretrain_image_embedding_dir,
            pretrain_text_embedding_dir = pretrain_text_embedding_dir,
            pretrain_image_dim = PRETRAIN_MULTIMODAL_DIM_DICT[multimodal_model][0],
            pretrain_text_dim = PRETRAIN_MULTIMODAL_DIM_DICT[multimodal_model][1],
            auto_schedule = True,
            gpu_device = args.gpu_device,
            cuda = args.cuda,
            seed = args.seed,
        )
        print('===================')
        print(exp_name)
        print('===================')
        run_model(config_dict)
