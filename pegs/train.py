import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from src.common import load_yaml, init_distributed_mode, get_rank, registry


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    args = parser.parse_args()
    
    return args


def setup_seeds(config):
    seed = config.run.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


def main(args):
    # config
    config = load_yaml(args.config)

    # distributed
    init_distributed_mode(config.run)
    # seed  [Required]
    setup_seeds(config)
    
    runner = registry.get_runner_class(config.run.runner)(config)
    runner.train()
    

if __name__ == "__main__":
    args = parse_args()

    main(args)
