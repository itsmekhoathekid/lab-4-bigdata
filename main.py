from utils import Solver, SparkConfig, DataLoader_stream
import torch
from models import LSTMWrapper, LSTM
import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Stream CIFAR dataset to Spark')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    return parser.parse_args()  # trả về namespace

if __name__ == '__main__':
    args = parse_args()  # đúng cú pháp

    config = load_config(args.config)
    config = config['stream_config']  # nếu YAML là stream: { ... }

    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    endless = config['endless']
    split = config['split']
    sleep_time = config['sleep']


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMWrapper(LSTM(), device=device)

    spark_config = SparkConfig()

    solver = Solver(
        model=model, 
        split=split, 
        spark_config=spark_config
    )

    solver.train()
