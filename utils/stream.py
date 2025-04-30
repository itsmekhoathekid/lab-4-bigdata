import time
import json
import socket
import argparse
import numpy as np
from tqdm import tqdm
import os
from torch.utils.data import DataLoader
from dataset import simDataset
import torch
import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

TCP_IP = "localhost"
TCP_PORT = 6100

class streaming:
    def __init__(self, dataset_path, batch_size, endless=False, split='train', sleep=3):
        self.datapath = dataset_path
        self.batch_size = batch_size
        self.endless = endless
        self.split = split
        self.sleep = sleep

    def dataloader_init(self):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = simDataset(datapath=self.datapath, device=device, split=self.split)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0)
        return dataloader

    def send_batch(self, tcp_connection):
        dataloader = self.dataloader_init()
        batch_id = 0
        pbar = tqdm(dataloader, desc="Streaming batches", unit="batch")

        for inputs, labels in pbar:
            x_batch = inputs.cpu().numpy().tolist()
            y_batch = labels.cpu().numpy().tolist()
    
            payload = {
                "x": x_batch,
                "y": y_batch
            }

            try:
                tcp_connection.send((json.dumps(payload) + "\n").encode())
            except BrokenPipeError:
                print("Connection closed")
                return
            except Exception as error_message:
                print(f"Send error: {error_message}")
                return

            batch_id += 1
            pbar.set_description(f"Sent batch #{batch_id}")
            time.sleep(self.sleep)


    def connectTCP(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((TCP_IP, TCP_PORT))
        s.listen(1)
        print(f"Waiting for connection on port {TCP_PORT}...")
        connection, address = s.accept()
        print(f"Connected to {address}")
        return connection, address

    def streamDataset(self, tcp_connection):
        self.send_batch(tcp_connection)



import yaml

def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def parse_args():
    parser = argparse.ArgumentParser(description='Stream CIFAR dataset to Spark')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    config = load_config(args.config)['stream_config']

    dataset_path = config['dataset_path']
    batch_size = config['batch_size']
    endless = config['endless']
    split = config['split']
    sleep_time = config['sleep']

    dataset = streaming(dataset_path, batch_size, endless, split, sleep_time)
    tcp_connection, _ = dataset.connectTCP()

    if endless:
        while True:
            dataset.streamDataset(tcp_connection)
    else:
        dataset.streamDataset(tcp_connection)

    tcp_connection.close()
