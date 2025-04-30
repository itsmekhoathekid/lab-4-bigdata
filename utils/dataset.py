from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch
from torch.utils.data import Dataset

class simDataset(Dataset):
    scaler = None  # Static scaler, shared between train and test

    def __init__(self, datapath=None, device=None, split=None, X=None, y=None):
        super().__init__()
        self.device = device

        if datapath is not None:
            df = pd.read_csv(datapath)
            self.x, self.y = self.transform(df, split)
        elif X is not None and y is not None:
            self.x = X
            self.y = y
        else:
            raise ValueError("Bạn phải cung cấp datapath hoặc cả X và y.")

    def transform(self, df, split):
        X = df.copy()
        Y = X['isFraud']
        X.drop(columns=['isFraud'], inplace=True)

        # Train/test split
        train_X, test_X, train_Y, test_Y = train_test_split(X, Y, test_size=0.2, random_state=42)

        if split == 'train':
            if simDataset.scaler is None:
                simDataset.scaler = MinMaxScaler()
                train_X_scaled = simDataset.scaler.fit_transform(train_X)
            else:
                train_X_scaled = simDataset.scaler.transform(train_X)
            return train_X_scaled, train_Y.values
        else:
            if simDataset.scaler is None:
                raise ValueError("Scaler chưa được fit. Vui lòng load tập train trước.")
            test_X_scaled = simDataset.scaler.transform(test_X)
            return test_X_scaled, test_Y.values

    def get_data(self):
        return self.x, self.y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        x = torch.tensor(self.x[idx], dtype=torch.float).to(self.device)
        y = torch.tensor(self.y[idx], dtype=torch.int64).to(self.device)
        return x, y





import numpy as np
from pyspark.context import SparkContext
from pyspark.sql.context import SQLContext
from pyspark.streaming.context import StreamingContext
from pyspark.streaming.dstream import DStream
from pyspark.ml.linalg import DenseVector
import json

class DataLoader_stream:
    def __init__(self, 
                 sparkContext:SparkContext, 
                 sparkStreamingContext: StreamingContext, 
                 sqlContext: SQLContext,
                 sparkConf) -> None:
        
        self.sc = sparkContext
        self.ssc = sparkStreamingContext
        self.sparkConf = sparkConf
        self.sql_context = sqlContext
        self.stream = self.ssc.socketTextStream(
            hostname=self.sparkConf.stream_host, 
            port=self.sparkConf.port
        )
    
    def parse_stream(self) -> DStream:
        json_stream = self.stream.map(lambda line: json.loads(line))
        stream = json_stream.map(lambda x: {"x": x["x"], "y": x["y"]})
        return stream
