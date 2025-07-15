# import pandas as pd

# # Đọc dữ liệu gốc
# df = pd.read_csv('sp500_industry.csv')

# # Đổi Symbol thành GOOG 
# df = df[df['Symbol'] == 'AMZN']

# # Chuẩn hoá cột Date: chỉ giữ phần ngày (YYYY-MM-DD)
# df['Date'] = pd.to_datetime(df['Date'], utc=True)

# # Sau đó chuyển sang định dạng ngày yyyy-mm-dd
# df['Date'] = df['Date'].dt.date
# # Chỉ giữ lại 2 cột: Date và Close
# df = df[['Date', 'Close']]

# # Lưu ra file mới
# df.to_csv('AMZN.csv', index=False)
# import pandas as pd

# # Bước 1: Đọc dữ liệu từ file CSV
# df = pd.read_csv('sp500_industry.csv')
# # Chuẩn hoá cột Date: chỉ giữ phần ngày (YYYY-MM-DD)
# df['Date'] = pd.to_datetime(df['Date'], utc=True)

# # Sau đó chuyển sang định dạng ngày yyyy-mm-dd
# df['Date'] = df['Date'].dt.date
# # Bước 2: Tìm sector của AMZN
# amzn_sector = df[df['Symbol'] == 'AMZN']['Sector'].unique()

# if len(amzn_sector) == 0:
#     print("Không tìm thấy mã cổ phiếu AMZN trong dữ liệu.")
# else:
#     amzn_sector = amzn_sector[0]  # Lấy giá trị sector

#     # Bước 3: Lọc các dòng có cùng sector
#     same_sector_df = df[df['Sector'] == amzn_sector]

#     # Bước 4: Lưu ra file mới
#     same_sector_df.to_csv('AMZN_industry.csv', index=False)

#     print(f"Đã lưu các cổ phiếu cùng ngành với AMZN ({amzn_sector}) vào file AMZN_industry.csv")

import pandas as pd

# Đọc file CSV
df = pd.read_csv('AMZN_industry.csv')

# Đếm số lượng symbol (giả sử cột tên là 'Symbol')
num_symbols = df['Symbol'].nunique()

print(f"Số lượng symbol trong file: {num_symbols}")
