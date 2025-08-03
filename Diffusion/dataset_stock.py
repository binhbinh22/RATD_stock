import pickle
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class StockDataset(Dataset):
    def __init__(self, csv_path, flag='train', size=None, data_mode='each_stock', symbol='GOOG', scale=True, reference = None, top_k = 1):
        """
        size: [seq_len, label_len, pred_len]
        data_mode: 'each_stock', 'all_stock', 'industry'
        symbol: tên của cổ phiếu (input luôn là GOOG)
        """
        if size is None:
            raise ValueError("cung cấp giá trị size, ví dụ: [30, 0, 7] (history 30 ngày, pred 7 ngày)")
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        self.data_mode = data_mode
        self.symbol = symbol
        self.scale = scale
        self.flag = flag
        self.top_k = top_k
        self.reference = reference


        df = pd.read_csv(csv_path, usecols=['Date', 'Close'])
        df.columns = df.columns.str.strip()
        
        df['Date'] = pd.to_datetime(df['Date'])
        
        df_input = df.sort_values('Date')

        # Lưu lại cột Date cho input
        self.dates = pd.to_datetime(df_input['Date'], utc=True)

        # Lấy giá đóng cửa của GOOG làm input
        data = df_input['Close'].values.reshape(-1, 1)
        self.raw_data = data.astype(np.float32)
        self.dim = self.raw_data.shape[1]


        total_len = len(self.raw_data)
        train = int(total_len * 0.7)
        
        # print(f"Train train {train}")
        # print(total_len)
        # print(top_k * self.pred_len * (total_len - 39))
        # print(len(reference))

        border1s = [0, int(total_len * 0.7), int(total_len * 0.7)] 
        border2s = [int(total_len * 0.7), total_len, total_len]
        border1s_reference = [0, int(total_len * 0.7) * top_k * self.pred_len, int(total_len * 0.7) * top_k]
        border2s_reference = [(int(total_len * 0.7) - self.seq_len - self.pred_len + 1) * top_k * self.pred_len, len(self.reference), len(self.reference)]
        type_map = {'train': 0, 'val': 1, 'test': 2}
        set_type = type_map[self.flag]
        border1 = border1s[set_type]
        border2 = border2s[set_type]
        border1_ref = border1s_reference[set_type]
        border2_ref = border2s_reference[set_type] 
        self.data = self.raw_data[border1:border2]
        self.reference = self.reference[border1_ref: border2_ref]

        self.scaler = MinMaxScaler()
        # self.scaler = StandardScaler() 
        # if self.scale:
        train_data = self.raw_data[0:border2s[0]]
        self.scaler.fit(train_data)
        #minmax
        self.min_value = self.scaler.data_min_
        self.max_value = self.scaler.data_max_
        
        self.data = (self.data - self.min_value) / (self.max_value - self.min_value)
        self.reference = (self.reference - self.min_value) / (self.max_value - self.min_value)
        #standard
        # self.mean, self.std = self.scaler.mean_, self.scaler.var_

        # self.data = self.scaler.transform(self.data)

        # if self.reference.ndim == 1:
        #     self.reference = self.reference.reshape(-1, 1)

        # self.reference = self.scaler.transform(self.reference)

        # if self.reference.shape[1] == 1:
        #     self.reference = self.reference.flatten()

        self.data_x = self.data[:,:]
        print(self.data.shape)
        self.data_y = self.data[:,:]
        
        
        
        #self.reference = torch.clamp(self.reference, min=0, max=self.data.shape[0] - self.seq_len - self.pred_len)
    def __getitem__(self, index):

        s_begin = index 
        s_end = s_begin + self.seq_len + self.pred_len
        # r_begin = s_end - self.label_len  # 20--40   40-->60
        r_begin = s_end - self.pred_len
        # r_end = r_begin + self.label_len + self.pred_len 
        r_end = r_begin + self.label_len + self.pred_len 
        reference = np.zeros((self.top_k * self.pred_len, self.dim))
    
        
        start_idx = (self.top_k * index) * self.pred_len #0 (0, 23)
        end_idx = (self.top_k * (index + 1)) * self.pred_len #24
        reference = self.reference[start_idx:end_idx]  # (24,)
        # print(segment)
        # print(self.mean, self.std)
        #print(reference)
        
            
                
        reference = reference.squeeze()
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        
        # Tạo mốc thời gian (timepoints) dựa trên index (có thể điều chỉnh tùy theo logic)
        timepoints = np.arange(self.seq_len + self.pred_len) * 1.0
        # Danh sách các feature id (số lượng feature = self.dim)
        feature_id = np.arange(self.dim) * 1.0
        
        # Tạo mask: observed_mask là mảng toàn 1
        observed_mask = np.ones_like(seq_x)
        # gt_mask được copy từ observed_mask nhưng phần dự báo (last pred_len) được đặt thành 0
        gt_mask = observed_mask.copy()
        gt_mask[-self.pred_len:] = 0.
        
        sample = {
            'observed_data': seq_x, 
            'observed_mask': observed_mask,
            'gt_mask': gt_mask,
            'timepoints': timepoints,
            'feature_id': feature_id,
            'reference': reference,
            #'reference_image': reference_image
        }
        return sample
    
    def __len__(self):
        return len(self.data) - self.seq_len - self.pred_len + 1
    
    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

def get_dataloader(csv_path, data_mode='all_stock', symbol='GOOG', size=None, batch_size=16, reference = None, top_k = 1):
    if size is None:
        size = [30, 0, 7]
    train_dataset = StockDataset(csv_path, flag='train', size=size, data_mode=data_mode, symbol=symbol, reference = reference, top_k = top_k)
    valid_dataset = StockDataset(csv_path, flag='val', size=size, data_mode=data_mode, symbol=symbol, reference = reference,  top_k = top_k)
    test_dataset = StockDataset(csv_path, flag='test', size=size, data_mode=data_mode, symbol=symbol, reference = reference,  top_k = top_k)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    print("Hello")
    
    return train_loader, valid_loader, test_loader



