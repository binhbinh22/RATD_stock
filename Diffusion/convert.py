import torch

tensor = torch.load("AAPL_k_n_only/AAPL_vs_only_1_1.pt")
# tensor = torch.load("k_n_industry_trend/GOOG_vs_industry_1_1.pt")
# tensor = torch.load("k_n_all_trend_old/GOOG_vs_all_trend_3_1.pt")
print(tensor)
print("Kích thước tensor:", tensor.shape)
print("Tổng số phần tử:", tensor.numel())

# import torch
# import os

# # Đường dẫn đến thư mục chứa các file .pt
# folder_paths = ['GOOG_k_n_all','GOOG_k_n_industry','GOOG_k_n_only']

# # Duyệt qua tất cả các file trong thư mục
# for folder_path in folder_paths:
#     for filename in os.listdir(folder_path):
#         if filename.endswith('.pt'):
#             file_path = os.path.join(folder_path, filename)

#             # Load tensor từ file
#             tensor = torch.load(file_path)

#             # Flatten tensor thành 1 chiều
#             flat_tensor = tensor.view(-1)

#             # Lưu lại tensor đã flatten, ghi đè file cũ
#             torch.save(flat_tensor, file_path)

#             print(f'Đã xử lý: {filename} - Shape mới: {flat_tensor.shape}')
