# xong GOOG
# xong GOOG
# sửa cho AMZN 9624
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset, DataLoader
import csv
import os
print("oke")
# -------------------------------
# TCN Encoder hỗ trợ (sử dụng Chomp1d và TemporalBlock)
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
# TCN Encoder định nghĩa theo RATD (cho dữ liệu GOOG)
# -------------------------------
class TCNEncoder(nn.Module):
    def __init__(self, input_size=1, num_channels=[16, 32], kernel_size=3, dropout=0.2, emb_dim=64):
        """
        input_size: Số kênh đầu vào (ở đây = 1 vì chỉ dùng dữ liệu của GOOG).
        num_channels: Danh sách số kênh của các lớp Conv1d, ví dụ [16, 32].
        emb_dim: Chiều của vector embedding cuối cùng.
        """
        super(TCNEncoder, self).__init__()
        layers = []
        in_channels = input_size
        for i, out_channels in enumerate(num_channels):
            dilation = 2 ** i
            padding = (kernel_size - 1) * dilation
            block = TemporalBlock(in_channels, out_channels, kernel_size,
                                    stride=1, dilation=dilation, padding=padding, dropout=dropout)
            layers.append(block)
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.proj = nn.Linear(in_channels, emb_dim)
    
    def forward(self, x):
        # x shape: (B, 1, seq_len)
        feat = self.network(x)         # (B, out_channels, seq_len)
        feat = torch.mean(feat, dim=2)   # Global average pooling -> (B, out_channels)
        emb = self.proj(feat)           # (B, emb_dim)
        return emb

# -------------------------------
# Decoder: hồi phục lại chuỗi đầu vào từ embedding
# -------------------------------
class Decoder(nn.Module):
    def __init__(self, emb_dim, window_size):
        super().__init__()
        self.fc = nn.Linear(emb_dim, window_size)
    
    def forward(self, x):
        # x: (B, emb_dim)
        out = self.fc(x)  # (B, window_size)
        return out.unsqueeze(1)  # (B, 1, window_size)

# -------------------------------
# Autoencoder: kết hợp Encoder và Decoder
# -------------------------------
class TCN_Autoencoder(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size, dropout, emb_dim, window_size):
        super(TCN_Autoencoder, self).__init__()
        self.encoder = TCNEncoder(input_size, num_channels, kernel_size, dropout, emb_dim)
        self.decoder = Decoder(emb_dim, window_size)
    
    def forward(self, x):
        emb = self.encoder(x)      # (B, emb_dim)
        recon = self.decoder(emb)  # (B, 1, window_size)
        return recon

# -------------------------------
# Dataset cho Pre-training với dữ liệu của GOOG
# -------------------------------
class GoDataTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, symbol="AMZN", window_size=96, scale=True):
        """
        csv_path: đường dẫn tới file CSV chứa các trường Symbol, Date, Close, Sector.
        symbol: dùng dữ liệu của một cổ phiếu (ví dụ: GOOG).
        window_size: độ dài của mỗi cửa sổ (ví dụ: 120 = seq_len + pred_len).
        scale: chuẩn hóa dữ liệu.
        """
        df = pd.read_csv(csv_path)
        # Lọc chỉ dữ liệu của cổ phiếu được chỉ định (GOOG)
        # df = df[df['Symbol'] == symbol]
        df['date'] = pd.to_datetime(df['date'], utc=True)
        # df = df.sort_values('Date')
        # Lấy cột Close
        prices = df['Close'].values.astype(np.float32)
        self.window_size = window_size
        self.scale = scale
        self.scaler = StandardScaler()
        if self.scale:
            prices = self.scaler.fit_transform(prices.reshape(-1, 1)).flatten()
        self.prices = prices
        # Tạo sliding windows
        self.windows = np.array([prices[i:i+window_size] for i in range(0, len(prices)-window_size+1)])
    
    def __len__(self):
        return len(self.windows)
    
    def __getitem__(self, index):
        window = self.windows[index]  # shape (window_size,)
        # Chuyển thành tensor với shape (1, window_size) (input_size = 1)
        return torch.tensor(window, dtype=torch.float32).unsqueeze(0)

# -------------------------------
# Tối ưu hóa hyperparameter và lưu kết quả
# -------------------------------
def hyperparameter_search(csv_path, symbol, window_size, device="cuda:0"):
    # Định nghĩa không gian tham số cần tối ưu
    import csv
    import os
    param_grid = {
        'num_channels': [    
            [32, 64],      
        ],
        'kernel_size': [3, 5],
        'dropout': [0.2, 0.3],
        'emb_dim': [32, 64],
        'batch_size': [16, 32],
        'learning_rate': [1e-4, 1e-3]
    }
    
    os.makedirs("results", exist_ok=True)

    # File kết quả CSV: theo symbol + window_size
    results_file = f"results/tcn_results_{symbol}_{window_size}.csv"
    write_header = not os.path.exists(results_file)

    # Nếu file chưa tồn tại → ghi header
    if write_header:
        with open(results_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([
                'num_channels', 'kernel_size', 'dropout', 'emb_dim', 
                'batch_size', 'learning_rate', 'epochs', 
                'train_loss', 'val_loss', 'best_epoch', 'best_val_loss'
            ])
    best_model_file = f"results/tcn_best_model_{symbol}_{window_size}.pt"
    best_encoder_file = f"results/tcn_best_encoder_{symbol}_{window_size}.pt"
    best_param_file = f"results/best_params_{symbol}_{window_size}.txt"
    # Tạo dataset
    dataset = GoDataTimeSeriesDataset(csv_path, symbol=symbol, window_size=window_size, scale=True)
    
    # Chia dataset thành train và validation (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    print(f"Dataset split: {len(train_dataset)} training samples, {len(val_dataset)} validation samples")
    
    # Biến để theo dõi model tốt nhất
    best_overall_loss = float('inf')
    best_model = None
    best_params = None
    
    # Thử nghiệm các tổ hợp tham số
    for num_channels in param_grid['num_channels']:
        for kernel_size in param_grid['kernel_size']:
            for dropout in param_grid['dropout']:
                for emb_dim in param_grid['emb_dim']:
                    for batch_size in param_grid['batch_size']:
                        for lr in param_grid['learning_rate']:
                            print(f"\n=== Testing parameters: num_channels={num_channels}, kernel_size={kernel_size}, "
                                  f"dropout={dropout}, emb_dim={emb_dim}, batch_size={batch_size}, lr={lr} ===")
                            
                            # Tạo dataloader cho tập train và validation
                            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                            
                            # Khởi tạo model
                            model = TCN_Autoencoder(
                                input_size=1,
                                num_channels=num_channels,
                                kernel_size=kernel_size,
                                dropout=dropout,
                                emb_dim=emb_dim,
                                window_size=window_size
                            ).to(device)
                            
                            optimizer = optim.Adam(model.parameters(), lr=lr)
                            criterion = nn.MSELoss()
                            
                            # Huấn luyện model
                            epochs = 50
                            best_val_loss = float('inf')
                            best_epoch = 0
                            best_model_state = None
                            
                            for epoch in range(1, epochs+1):
                                # Training
                                model.train()
                                train_loss = 0.0
                                for batch in train_loader:
                                    batch = batch.to(device)
                                    optimizer.zero_grad()
                                    recon = model(batch)
                                    loss = criterion(recon, batch)
                                    loss.backward()
                                    optimizer.step()
                                    train_loss += loss.item() * batch.size(0)
                                train_loss /= len(train_dataset)
                                
                                # Validation
                                model.eval()
                                val_loss = 0.0
                                with torch.no_grad():
                                    for batch in val_loader:
                                        batch = batch.to(device)
                                        recon = model(batch)
                                        loss = criterion(recon, batch)
                                        val_loss += loss.item() * batch.size(0)
                                val_loss /= len(val_dataset)
                                
                                print(f"Epoch {epoch}/{epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
                                
                                # Lưu model tốt nhất trong lần thử nghiệm này theo validation loss
                                if val_loss < best_val_loss:
                                    best_val_loss = val_loss
                                    best_epoch = epoch
                                    best_model_state = model.state_dict().copy()
                            
                            # Khôi phục model về trạng thái tốt nhất (có val_loss nhỏ nhất)
                            model.load_state_dict(best_model_state)
                            
                            # Lưu kết quả vào CSV
                            with open(results_file, 'a', newline='') as file:
                                writer = csv.writer(file)
                                writer.writerow([
                                    str(num_channels), kernel_size, dropout, emb_dim, 
                                    batch_size, lr, epochs, train_loss, val_loss, best_epoch, best_val_loss
                                ])
                            
                            # Cập nhật model tốt nhất tổng thể dựa trên validation loss
                            if best_val_loss < best_overall_loss:
                                best_overall_loss = best_val_loss
                                best_model = model
                                best_params = {
                                    'num_channels': num_channels,
                                    'kernel_size': kernel_size,
                                    'dropout': dropout,
                                    'emb_dim': emb_dim,
                                    'batch_size': batch_size,
                                    'learning_rate': lr,
                                    'best_epoch': best_epoch,
                                    'best_val_loss': best_val_loss
                                }
                                
                                # Lưu model tốt nhất
                                torch.save(model.state_dict(), best_model_file)
                                torch.save(model.encoder.state_dict(), best_encoder_file)
                                
                                # Lưu thông tin tham số tốt nhất
                                with open(results_file, 'w') as f:
                                    for key, value in best_params.items():
                                        f.write(f"{key}: {value}\n")
    
    print("\nHyperparameter optimization completed!")
    print(f"Best parameters: {best_params}")
    print(f"Best validation loss: {best_overall_loss:.6f}")
    print(f"Results saved to {results_file}")
    print(f"Best model saved to results/tcn_autoencoder_best.pt")
    print(f"Best encoder saved to results/tcn_encoder_best.pt")
    
    return best_model, best_params

# -------------------------------
# Hàm main để chạy tối ưu hóa hyperparameter
# -------------------------------
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol')
parser.add_argument('--window_size', type=int, default=96, help='Sliding window size')
args = parser.parse_args()

if __name__ == "__main__":
    csv_path = f"/home/oem/ntthu/RATD_stock/RATD_stock/{args.symbol}.csv"
    symbol = args.symbol
    window_size = args.window_size
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting hyperparameter search for {symbol} with window_size={window_size}")
    print(f"Using device: {device}")
    
    best_model, best_params = hyperparameter_search(
        csv_path=csv_path,
        symbol=symbol,
        window_size=window_size,
        device=device
    )
