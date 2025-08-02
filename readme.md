1. Chạy train_TCN
- Thay đổi dataset, seq, predict
2. chuyển model tcn sang retrieval 
3. chuyển retrieval sang diffusion 
python exe_stock_forecasting_only.py --data_mode each_stock --symbol AAPL --seed 2024 --device cuda:0 --csv_path 'AAPL.csv' --top_k 1 3 5 10 20 &
python exe_stock_forecasting_only.py --data_mode each_stock --symbol AAPL --seed 2025 --device cuda:0 --csv_path 'AAPL.csv' --top_k 1 3 5 10 20 &
python exe_stock_forecasting_only.py --data_mode each_stock --symbol AAPL --seed 2026 --device cuda:0 --csv_path 'AAPL.csv' --top_k 1 3 5 10 20 

