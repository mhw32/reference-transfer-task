import os
from copy import deepcopy
from reference.agents import TrainAgent
from reference.setup import process_config
from reference.setup import load_json


def run(config_path):
    config = process_config(config_path)
    agent = TrainAgent(config)

    agent.run()
    agent.finalise()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    args = parser.parse_args()

    run(args.config)