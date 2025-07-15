
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import csv
import copy
# -------------------------------
# Các lớp hỗ trợ cho TCN: Chomp1d & TemporalBlock
# -------------------------------
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[..., :-self.chomp_size] if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

# -------------------------------
# TCN Encoder
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: số kênh đầu vào = số cổ phiếu (num_stocks)
        num_channels: danh sách số kênh cho các tầng Conv1d (ví dụ: [16, 32])
        emb_dim: chiều vector embedding cuối cùng
        """
        super(TCNEncoder, self).__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation_size = 2 ** i
            padding = (kernel_size - 1) * dilation_size
            block = TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                    dilation=dilation_size, padding=padding, dropout=dropout)
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)
    
    def forward(self, x):
        # x: (B, input_size, seq_len)
        feat = self.network(x)         # (B, out_channels, seq_len)
        feat = torch.mean(feat, dim=2)   # Global average pooling -> (B, out_channels)
        emb = self.proj(feat)           # (B, emb_dim)
        return emb

# -------------------------------
# Decoder: Dự đoán lại toàn bộ window
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, emb_dim, window_size, num_stocks):  # Thêm num_stocks
        super(Decoder, self).__init__()
        self.fc = nn.Linear(emb_dim, num_stocks * window_size)
        self.window_size = window_size
        self.num_stocks = num_stocks
    
    def forward(self, x):
        # x: (B, emb_dim)
        out = self.fc(x)  # (B, num_stocks * window_size)
        return out.reshape(-1, self.num_stocks, self.window_size)  # (B, num_stocks, window_size)


# -------------------------------
# Autoencoder: Encoder + Decoder
# -------------------------------
class TCN_Autoencoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, emb_dim, window_size):
        super(TCN_Autoencoder, self).__init__()
        self.encoder = TCNEncoder(input_size, num_channels, kernel_size, dropout, emb_dim)
        self.decoder = Decoder(emb_dim, window_size, input_size)  # input_size = num_stocks
    
    def forward(self, x):
        emb = self.encoder(x)      # (B, emb_dim)
        recon = self.decoder(emb)  # (B, num_stocks, window_size)
        return recon

param_grid = {
    'num_channels': [[32, 64]],
    'kernel_size': [3,5],
    'dropout': [0.1,0.2],
    'emb_dim': [32,64],
    'batch_size': [16,32],
    'learning_rate': [1e-4,1e-3]
}

# -------------------------------
# Dataset cho Pre-training toàn bộ dữ liệu time series
# -------------------------------
class AllStockTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, scale=True, window_size=96):
        df = pd.read_csv(csv_path)
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
        
        # SỬA: Thay fillna method bằng ffill/bfill
        pivot_df = pivot_df.ffill().bfill()
        
        self.data = pivot_df.values.astype(np.float32)
        self.window_size = window_size
        self.scale = scale
        self.scaler = StandardScaler()
        if self.scale:
            self.data = self.scaler.fit_transform(self.data)
        
        T = self.data.shape[0]
        windows = []
        for i in range(0, T - window_size + 1):
            windows.append(self.data[i:i+window_size])
        self.windows = np.array(windows)
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        window = self.windows[index]
        return torch.tensor(window, dtype=torch.float32).permute(1, 0)


import os
os.makedirs('results', exist_ok=True)
# Sửa lại phần khởi tạo dataset và chia tập train/validation
csv_path = "sp500_industry.csv"
window_size = 96
device = "cuda:0" if torch.cuda.is_available() else "cpu"
dataset = AllStockTimeSeriesDataset(csv_path, scale=True, window_size=window_size)
num_stocks = dataset.data.shape[1]  # Số lượng cổ phiếu
print(num_stocks)
# Chia dataset thành train và validation (80% train, 20% validation)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
)
print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")

# Sửa file kết quả để thêm cột validation loss
results_file = "tcn_all_results_9624.csv"
with open(results_file, 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['num_channels', 'kernel_size', 'dropout', 'emb_dim', 
                    'batch_size', 'learning_rate', 'best_epoch', 'train_loss', 'val_loss'])

# Sửa hàm train_evaluate_model để tính validation loss
def train_evaluate_model(num_channels, kernel_size, dropout, emb_dim, batch_size, lr):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    model = TCN_Autoencoder(
        input_size=num_stocks,
        num_channels=num_channels,
        kernel_size=kernel_size,
        dropout=dropout,
        emb_dim=emb_dim,
        window_size=window_size
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # SỬA: Khởi tạo best_model_state ngay từ đầu
    best_val_loss = float('inf')
    best_epoch = 0
    best_train_loss = 0
    best_model_state = copy.deepcopy(model.state_dict())  # THÊM DÒNG NÀY
    
    epochs = 50
    
    for epoch in range(1, epochs+1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            recon = model(batch)
            loss = criterion(recon, batch)
            loss.backward()
            
            # SỬA: Thêm gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item() * batch.size(0)
        train_loss /= len(train_dataset)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                recon = model(batch)
                loss = criterion(recon, batch)
                val_loss += loss.item() * batch.size(0)
        val_loss /= len(val_dataset)
        
        print(f"Params: {num_channels}, {kernel_size}, {dropout}, {emb_dim}, {batch_size}, {lr} - "
              f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_train_loss = train_loss
            best_model_state = copy.deepcopy(model.state_dict())  # SỬA: Dùng copy.deepcopy
    
    model.load_state_dict(best_model_state)
    
    return model, best_epoch, best_train_loss, best_val_loss

# Phần grid search - KHÔNG SỬA
best_model = None
best_params = None
best_overall_val_loss = float('inf')

for num_channels in param_grid['num_channels']:
    for kernel_size in param_grid['kernel_size']:
        for dropout in param_grid['dropout']:
            for emb_dim in param_grid['emb_dim']:
                for batch_size in param_grid['batch_size']:
                    for lr in param_grid['learning_rate']:
                        print(f"\nTesting parameters: num_channels={num_channels}, kernel_size={kernel_size}, "
                              f"dropout={dropout}, emb_dim={emb_dim}, batch_size={batch_size}, lr={lr}")
                        
                        model, best_epoch, best_train_loss, best_val_loss = train_evaluate_model(
                            num_channels, kernel_size, dropout, emb_dim, batch_size, lr
                        )
                        
                        with open(results_file, 'a', newline='') as file:
                            writer = csv.writer(file)
                            writer.writerow([str(num_channels), kernel_size, dropout, emb_dim, 
                                            batch_size, lr, best_epoch, best_train_loss, best_val_loss])
                        
                        if best_val_loss < best_overall_val_loss:
                            best_overall_val_loss = best_val_loss
                            best_params = {
                                'num_channels': num_channels,
                                'kernel_size': kernel_size,
                                'dropout': dropout,
                                'emb_dim': emb_dim,
                                'batch_size': batch_size,
                                'learning_rate': lr,
                                'best_epoch': best_epoch,
                                'best_train_loss': best_train_loss,
                                'best_val_loss': best_val_loss
                            }
                            best_model = model
                            torch.save(model.state_dict(), "results/tcn_all_best_9624.pt")
                            torch.save(model.encoder.state_dict(), "results/tcn_all_encoder_best_9624.pt")
                            
                            with open("results/best_params_all_9624.txt", 'w') as f:
                                for key, value in best_params.items():
                                    f.write(f"{key}: {value}\n")

print(f"Best parameters: {best_params}")
print(f"Best validation loss: {best_overall_val_loss}")
print(f"Results saved to {results_file}")
print(f"Best model saved to tcn_all_best.pt")
print(f"Best encoder saved to tcn_all_encoder_best.pt")