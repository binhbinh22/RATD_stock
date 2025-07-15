import pandas as pd
import numpy as np

# Đọc và kiểm tra CSV
df = pd.read_csv('AMZN_industry.csv')
print(f"CSV shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"Data types:\n{df.dtypes}")

# Kiểm tra missing values
print(f"Missing values:\n{df.isnull().sum()}")

# Kiểm tra duplicate dates
print(f"Duplicate dates: {df['Date'].duplicated().sum()}")

# Kiểm tra range của Close prices
print(f"Close price range: {df['Close'].min()} - {df['Close'].max()}")

# Kiểm tra có giá trị âm không
if (df['Close'] < 0).any():
    print("Warning: Negative prices found!")


# 1. Kiểm tra unique symbols
print(f"Unique symbols: {df['Symbol'].nunique()}")
print(f"Symbol list: {df['Symbol'].unique()}")

# 2. Kiểm tra unique dates
print(f"Unique dates: {df['Date'].nunique()}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# 3. Kiểm tra combination Symbol + Date
print(f"Unique Symbol-Date combinations: {df[['Symbol', 'Date']].drop_duplicates().shape[0]}")

# 4. Tìm duplicate records thực sự
duplicates = df[df.duplicated(subset=['Symbol', 'Date'], keep=False)]
print(f"Duplicate Symbol-Date pairs: {len(duplicates)}")

# 5. Xem một vài ví dụ duplicate
if len(duplicates) > 0:
    print("\nSample duplicates:")
    print(duplicates.head(10))

# 6. Kiểm tra format Date
print(f"\nSample dates: {df['Date'].head()}")

