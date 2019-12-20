"""
Script to run all files below for a dataset.
"""

import os
import subprocess

text_scripts = [
    'huggingface.py',
    'infersent.py',
    'skipthought.py',
    'word2vec.py',
    'glove.py'
]
image_scripts = [
    'vgg.py',
    'resnet.py',
    'ir_imagenet.py',
    'la_imagenet.py',
    'vae.py',
]
multimodal_scripts = [
    'ir_coco.py',
    'vaegan.py',
    'vaevae.py',
]

cur_dir = os.path.dirname(__file__)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--context-condition', type=str, default='all', 
                        choices=['all', 'far', 'close'])
    parser.add_argument('--split-mode', type=str, default='easy',
                        choices=['easy', 'hard'])
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    # --- 
    parser.add_argument('--image-only', action='store_true', default=False)
    parser.add_argument('--text-only', action='store_true', default=False)
    args = parser.parse_args()
    
    cuda_str = '--cuda' if args.cuda else ''
    
    def make_command(name):
        if name == 'huggingface.py':
            extra = '--all'
        else:
            extra = ''
        return f'python {os.path.join(cur_dir, name)} {args.dataset} {args.data_dir} --context-condition {args.context_condition} --split-mode {args.split_mode} --gpu-device {args.gpu_device} {cuda_str} --seed {args.seed} {extra}'
    
    if args.image_only:
        names = image_scripts
    elif args.text_only:
        names = text_scripts
    else:
        names = text_scripts + image_scripts + multimodal_scripts

    for i, name in enumerate(names):
        command = make_command(name)
        print(f"Running command ({i+1}/{len(names)}): {command}")
        process = subprocess.Popen(command, shell=True, executable='/bin/bash')
        process.wait()  # sequential computation
