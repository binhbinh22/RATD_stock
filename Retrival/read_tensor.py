import torch

# tensor = torch.load("k_n_industry/GOOG_vs_industry_1_1.pt")
# tensor = torch.load("k_n_industry_trend/GOOG_vs_industry_1_1.pt")
# tensor = torch.load("AMZN_k_n_industry/AMZN_vs_industry_3_10.pt")
tensor = torch.load("GOOG_k_n_only/GOOG_vs_only_3_10.pt")
print(tensor)
print("Kích thước tensor:", tensor.shape)
print("Tổng số phần tử:", tensor.numel())