import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler

class LSTMForecaster(nn.Module):
    """LSTM-based time series forecasting model"""
    
    def __init__(self, input_size=1, hidden_size=128, num_layers=2, output_size=1, dropout=0.2):
        super(LSTMForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

class TimeSeriesDataset(Dataset):
    def __init__(self, data, lookback=30):
        self.data = data
        self.lookback = lookback
        
    def __len__(self):
        return len(self.data) - self.lookback
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.lookback]
        y = self.data[idx+self.lookback]
        return torch.FloatTensor(x), torch.FloatTensor([y])

class LSTMTrainer:
    def __init__(self, model, lr=0.001):
        self.model = model
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        
    def train(self, train_loader, epochs=50):
        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            for x_batch, y_batch in train_loader:
                self.optimizer.zero_grad()
                y_pred = self.model(x_batch.unsqueeze(-1))
                loss = self.criterion(y_pred, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(train_loader):.4f}')
    
    def predict(self, data, lookback=30):
        self.model.eval()
        with torch.no_grad():
            x = torch.FloatTensor(data[-lookback:]).unsqueeze(0).unsqueeze(-1)
            prediction = self.model(x)
        return prediction.item()
