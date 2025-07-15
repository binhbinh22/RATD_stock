import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import copy
import math
import itertools

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TCNEncoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, emb_dim=64):
        super(TCNEncoder, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(num_channels[-1], emb_dim)

    def forward(self, x):
        # x shape: (batch_size, sequence_length, num_features)
        x = x.transpose(1, 2)  # (batch_size, num_features, sequence_length)
        
        # Debug: Check input
        if torch.isnan(x).any():
            print("NaN detected in TCNEncoder input!")
            return None
            
        x = self.network(x)
        
        # Debug: Check after network
        if torch.isnan(x).any():
            print("NaN detected after TCN network!")
            return None
            
        x = self.global_pool(x)  # (batch_size, num_channels[-1], 1)
        x = x.squeeze(-1)  # (batch_size, num_channels[-1])
        
        # Debug: Check after pooling
        if torch.isnan(x).any():
            print("NaN detected after global pooling!")
            return None
            
        x = self.fc(x)  # (batch_size, emb_dim)
        
        # Debug: Check final output
        if torch.isnan(x).any():
            print("NaN detected in final encoder output!")
            return None
            
        return x

class TCNDecoder(nn.Module):
    def __init__(self, emb_dim, output_size):
        super(TCNDecoder, self).__init__()
        self.fc = nn.Linear(emb_dim, output_size)

    def forward(self, x):
        # Debug: Check input
        if torch.isnan(x).any():
            print("NaN detected in TCNDecoder input!")
            return None
            
        x = self.fc(x)
        
        # Debug: Check output
        if torch.isnan(x).any():
            print("NaN detected in TCNDecoder output!")
            return None
            
        return x

class TCNAutoencoder(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, emb_dim=64, window_size=96):
        super(TCNAutoencoder, self).__init__()
        self.encoder = TCNEncoder(num_inputs, num_channels, kernel_size, dropout, emb_dim)
        self.decoder = TCNDecoder(emb_dim, window_size * num_inputs)
        self.window_size = window_size
        self.num_inputs = num_inputs

    def forward(self, x):
        # Debug: Check input
        if torch.isnan(x).any():
            print("NaN detected in TCNAutoencoder input!")
            print(f"Input shape: {x.shape}")
            print(f"Input stats: min={x.min()}, max={x.max()}, mean={x.mean()}")
            return None
            
        encoded = self.encoder(x)
        if encoded is None:
            return None
            
        decoded = self.decoder(encoded)
        if decoded is None:
            return None
            
        # Reshape to match input
        batch_size = x.size(0)
        decoded = decoded.view(batch_size, self.window_size, self.num_inputs)
        
        return decoded

class AllStockTimeSeriesDataset(Dataset):
    def __init__(self, csv_path, window_size=96):
        self.window_size = window_size
        print(f"Loading data from {csv_path}...")
        
        # Load and check data
        df = pd.read_csv(csv_path)
        print(f"Raw data shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Data types:\n{df.dtypes}")
        
        # Check for missing values
        missing_values = df.isnull().sum()
        print(f"Missing values:\n{missing_values}")
        
        # Check data range
        print(f"Close price range: {df['Close'].min()} - {df['Close'].max()}")
        
        # Pivot data
        print("Pivoting data...")
        try:
            pivot_df = df.pivot(index='Date', columns='Symbol', values='Close')
            print(f"Pivot shape: {pivot_df.shape}")
        except Exception as e:
            print(f"Pivot error: {e}")
            raise
        
        # Check pivot result
        print(f"Pivot NaN count: {pivot_df.isnull().sum().sum()}")
        
        # Handle missing values
        print("Handling missing values...")
        pivot_df = pivot_df.fillna(method='ffill').fillna(method='bfill')
        
        # Final check after fillna
        remaining_nan = pivot_df.isnull().sum().sum()
        print(f"Remaining NaN after fillna: {remaining_nan}")
        
        if remaining_nan > 0:
            print("Warning: Still have NaN values, dropping them...")
            pivot_df = pivot_df.dropna()
        
        # Convert to numpy
        self.data = pivot_df.values
        print(f"Final data shape: {self.data.shape}")
        print(f"Data range before scaling: {self.data.min()} - {self.data.max()}")
        
        # Check for inf values
        inf_count = np.isinf(self.data).sum()
        print(f"Inf values: {inf_count}")
        
        if inf_count > 0:
            print("Replacing inf values with nan and dropping...")
            self.data = np.where(np.isinf(self.data), np.nan, self.data)
            # Remove rows with nan
            mask = ~np.isnan(self.data).any(axis=1)
            self.data = self.data[mask]
            print(f"Data shape after removing inf: {self.data.shape}")
        
        # Scale data
        print("Scaling data...")
        self.scaler = StandardScaler()
        
        # Check variance before scaling
        variances = np.var(self.data, axis=0)
        zero_var_cols = np.where(variances == 0)[0]
        if len(zero_var_cols) > 0:
            print(f"Warning: Columns with zero variance: {zero_var_cols}")
        
        self.data = self.scaler.fit_transform(self.data)
        
        # Final checks
        print(f"Data range after scaling: {self.data.min()} - {self.data.max()}")
        print(f"Data mean: {self.data.mean()}, std: {self.data.std()}")
        
        nan_after_scale = np.isnan(self.data).sum()
        inf_after_scale = np.isinf(self.data).sum()
        print(f"NaN after scaling: {nan_after_scale}")
        print(f"Inf after scaling: {inf_after_scale}")
        
        if nan_after_scale > 0 or inf_after_scale > 0:
            raise ValueError("Data contains NaN or Inf after preprocessing!")
        
        # Calculate number of windows
        self.num_windows = len(self.data) - self.window_size + 1
        print(f"Number of windows: {self.num_windows}")
        
        if self.num_windows <= 0:
            raise ValueError(f"Not enough data for window size {window_size}")

    def __len__(self):
        return self.num_windows

    def __getitem__(self, idx):
        if idx >= self.num_windows:
            raise IndexError(f"Index {idx} out of range for {self.num_windows} windows")
            
        start_idx = idx
        end_idx = start_idx + self.window_size
        
        window = self.data[start_idx:end_idx]
        tensor = torch.FloatTensor(window)
        
        # Debug check
        if torch.isnan(tensor).any():
            print(f"NaN found in sample {idx}!")
            print(f"Window range: {start_idx}:{end_idx}")
            print(f"Window stats: min={window.min()}, max={window.max()}")
            
        if torch.isinf(tensor).any():
            print(f"Inf found in sample {idx}!")
            
        return tensor

def train_evaluate_model(model, train_loader, val_loader, params, num_epochs=50):
    print(f"\nTraining with params: {params}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
    
    # Initialize
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=params['learning_rate'])
    
    # IMPORTANT: Initialize best_model_state immediately
    best_model_state = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Debug first batch
            if epoch == 0 and batch_idx == 0:
                print(f"First batch shape: {batch.shape}")
                print(f"First batch stats: min={batch.min()}, max={batch.max()}, mean={batch.mean()}")
                print(f"First batch NaN: {torch.isnan(batch).any()}")
                print(f"First batch Inf: {torch.isinf(batch).any()}")
            
            optimizer.zero_grad()
            
            # Forward pass
            reconstructed = model(batch)
            
            if reconstructed is None:
                print(f"Model returned None at epoch {epoch}, batch {batch_idx}")
                return model, epoch, float('nan'), float('nan')
            
            loss = criterion(reconstructed, batch)
            
            # Check loss
            if torch.isnan(loss):
                print(f"NaN loss at epoch {epoch}, batch {batch_idx}")
                print(f"Reconstructed stats: min={reconstructed.min()}, max={reconstructed.max()}")
                return model, epoch, float('nan'), float('nan')
            
            # Backward pass with gradient clipping
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                reconstructed = model(batch)
                if reconstructed is None:
                    return model, epoch, float('nan'), float('nan')
                    
                loss = criterion(reconstructed, batch)
                if torch.isnan(loss):
                    return model, epoch, float('nan'), float('nan')
                    
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        # Print progress
        print(f"Epoch {epoch+1}/{num_epochs}: train_loss={avg_train_loss:.6f}, val_loss={avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            print(f"✓ New best model saved (improvement: {best_val_loss - avg_val_loss:.6f})")
            best_val_loss = avg_val_loss
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
    
    # Load best model
    print(f"Loading best model from epoch {best_epoch}")
    model.load_state_dict(best_model_state)
    
    return model, best_epoch, avg_train_loss, best_val_loss

# Main execution
if __name__ == "__main__":
    # Simple parameters for testing
    param_grid = {
        'num_channels': [[16, 32], [32, 64]],
        'kernel_size': [3,5],
        'dropout': [0.1,0.2],
        'emb_dim': [32,64],
        'batch_size': [16,32],
        'learning_rate': [1e-4,1e-5]
    }
    
    csv_path = 'AMZN_industry.csv'  # Thay đổi path này
    window_size = 96
    
    try:
        # Load dataset
        dataset = AllStockTimeSeriesDataset(csv_path, window_size=window_size)
        
        # Split dataset
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        print(f"Dataset split: train={train_size}, val={val_size}")
        
        # Test one parameter combination
        params = {
            'num_channels': [8, 16],
            'kernel_size': 3,
            'dropout': 0.1,
            'emb_dim': 16,
            'batch_size': 16,
            'learning_rate': 1e-4
        }
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False, num_workers=0)
        
        # Test first batch
        print("\nTesting first batch...")
        for batch in train_loader:
            print(f"Batch shape: {batch.shape}")
            print(f"Batch stats: min={batch.min()}, max={batch.max()}, mean={batch.mean()}")
            break
        
        # Create model
        num_features = dataset.data.shape[1]
        model = TCNAutoencoder(
            num_inputs=num_features,
            num_channels=params['num_channels'],
            kernel_size=params['kernel_size'],
            dropout=params['dropout'],
            emb_dim=params['emb_dim'],
            window_size=window_size
        )
        
        print(f"Model created with {num_features} input features")
        
        # Train model
        model, best_epoch, best_train_loss, best_val_loss = train_evaluate_model(
            model, train_loader, val_loader, params, num_epochs=10  # Reduced for testing
        )
        
        print(f"Training completed. Best epoch: {best_epoch}, Best val loss: {best_val_loss}")
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
