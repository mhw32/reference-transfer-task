"""
Vary the amount of training data that is given to training
the witness function. Everything else remains the same.
"""

import os
import json
from dotmap import DotMap
from copy import deepcopy
from reference.agents import TrainAgent
from reference.setup import process_config
from reference.setup import load_json

DATA_SIZES = [0.005, 0.01, 0.05, 0.1, 0.25]  # , 0.5]
DATA_SIZES = DATA_SIZES[::-1]


def run(config_path, data_size):
    with open(config_path) as fp:
        old_config = json.load(fp)
        old_data = old_config['data']
        old_data['data_size'] = data_size
        old_exp_name = old_config["exp_name"]
        old_optim = old_config['optim']
        old_optim['auto_schedule'] = False

    config = process_config(
        config_path, 
        override_dotmap = {
            "exp_name": f'{old_exp_name}_weaksup_{data_size}',
            "data": DotMap(old_data),
            "optim": DotMap(old_optim),
        },
    )
    agent = TrainAgent(config)

    agent.run()
    agent.finalise()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='path to config file')
    args = parser.parse_args()

    for data_size in DATA_SIZES:
        run(args.config, data_size)
