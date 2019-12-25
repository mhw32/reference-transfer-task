import os
import json
import numpy as np

from reference.agents import EvaluateAgent


def eval_model(checkpoint_dir, checkpoint_name):
    agent = EvaluateAgent(checkpoint_dir, checkpoint_name=checkpoint_name)
    agent.run()
    agent.finalise()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str, choices=['chairs_in_context', 'colors_in_context', 'colorgrids_in_context'])
    parser.add_argument('--gpu-device', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    exp_base = '/mnt/fs5/wumike/reference/trained_models/12_22'
    exp_folders = os.listdir(os.path.join(exp_base, 'experiments'))
    exp_folders = [exp_folder for exp_folder in exp_folders if args.dataset in exp_folder]

    for exp_folder in exp_folders:
        exp_full_folder = os.path.join(exp_base, 'experiments', exp_folder)
        exp_date_folder = os.listdir(exp_full_folder)
        exp_date_folder = exp_date_folder[-1]
        checkpoint_dir = os.path.join(exp_full_folder, exp_date_folder)
        checkpoint_name = 'model_best.pth.tar'

        eval_model(checkpoint_dir, checkpoint_name)
