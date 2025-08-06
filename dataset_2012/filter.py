import pandas as pd

df = pd.read_csv('/home/user11/binhnkt/data_new/all.csv')

df_out = df[df['Symbol'] == 'LIN']

output_path = '/home/user11/binhnkt/data_new/LIN.csv'
df_out.to_csv(output_path, index=False)

print(f' Đã tạo file {output_path}')
