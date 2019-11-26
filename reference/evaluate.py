import os
from copy import deepcopy
from reference.agents import TrainAgent, EvaluateAgent
from reference.setup import process_config
from reference.setup import load_json


def run(checkpoint_dir, checkpoint_name):
    agent = EvaluateAgent(checkpoint_dir, checkpoint_name=checkpoint_name)
    agent.run()
    agent.finalise()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint_dir', type=str, default='path to trained checkpoint')
    parser.add_argument('--checkpoint-name', type=str, default='model_best.pth.tar',
                        choices=['model_best.pth.tar', 'checkpoint.pth.tar'])
    args = parser.parse_args()

    run(args.checkpoint_dir, args.checkpoint_name)
