import torch.nn as nn
import torch
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=128, num_layers=4, num_classes=2, bidirectional=True):
        
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(2 * hidden_size if bidirectional else hidden_size, num_classes)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.model_name = 'LSTM'
        self.bidirectional = bidirectional
    def forward(self, x):
        # Khởi tạo hidden state và cell state
        # x shape: (B, T, input_size)
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))  # out shape: (B, T, hidden_size)
        out = out[:, -1, :]              # Lấy output ở time step cuối
        out = self.fc(out)

        
        return out

    def forward_embeddings(self, x):
        """
        Trả về đặc trưng từ tầng LSTM, dùng cho Spectral Signature Defense.
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)
            print("x shape l2 : ", x.shape)
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1), x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return out[:, -1, :]  # (batch_size, hidden_size * num_directions)

import numpy as np

class LSTMWrapper:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

    def train(self, df):
        self.model.train()

        features = np.stack(df.select("x").rdd.map(lambda row: row['x']).collect())
        labels = np.array(df.select("y").rdd.map(lambda row: row['y']).collect())

        x = torch.tensor(features, dtype=torch.float32).to(self.device)  # [batch_size, seq_len=7]
        x = x.squeeze(0).unsqueeze(-1)
        y = torch.tensor(labels, dtype=torch.long).to(self.device)

        print("x shape: ", x.shape)
        print("y shape: ", y.shape)
        y = y.view(-1)
        outputs = self.model(x)
        loss = self.criterion(outputs, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == y).float().mean().item()
        precision = recall = f1 = accuracy  # tạm placeholder

        return predictions.tolist(), accuracy, precision, recall, f1
