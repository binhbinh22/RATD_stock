import numpy as np
import torch
from torch.optim import Adam
from tqdm import tqdm
import pickle
import matplotlib.pyplot as plt

def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=3,
    foldername="",
):
    optimizer = Adam(model.parameters(), lr=config["lr"], weight_decay=1e-6)
    if foldername != "":
        output_path = foldername + "/model.pth"

    p1 = int(0.75 * config["epochs"])
    p2 = int(0.9 * config["epochs"])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    # Early Stopping variables
    best_valid_loss = 1e10
    patience = 10
    patience_counter = 0
    
    for epoch_no in range(config["epochs"]):
        avg_loss = 0
        model.train()
        with tqdm(train_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, train_batch in enumerate(it, start=1):
                optimizer.zero_grad()

                loss = model(train_batch)
                loss.backward()
                avg_loss += loss.item()
                optimizer.step()
                it.set_postfix(
                    ordered_dict={
                        "avg_epoch_loss": avg_loss / batch_no,
                        "epoch": epoch_no,
                    },
                    refresh=False,
                )
                if batch_no >= config["itr_per_epoch"]:
                    break

        lr_scheduler.step()
        
        # Validation và Early Stopping
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                with tqdm(valid_loader, mininterval=5.0, maxinterval=50.0) as it:
                    for batch_no, valid_batch in enumerate(it, start=1):
                        loss = model(valid_batch, is_train=0)
                        avg_loss_valid += loss.item()
                        it.set_postfix(
                            ordered_dict={
                                "valid_avg_epoch_loss": avg_loss_valid / batch_no,
                                "epoch": epoch_no,
                            },
                            refresh=False,
                        )
            
            # Tính validation loss trung bình
            avg_loss_valid = avg_loss_valid / len(valid_loader)
            
            # Kiểm tra Early Stopping
            if avg_loss_valid < best_valid_loss:
                best_valid_loss = avg_loss_valid
                patience_counter = 0
                # Lưu model tốt nhất
                if foldername != "":
                    torch.save(model.state_dict(), output_path)
                print(
                    f"\nBest loss updated to {avg_loss_valid:.6f} at epoch {epoch_no}"
                )
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter} validation checks")
                
                # Early stopping
                if patience_counter >= patience:
                    print(f"Early stopping triggered at epoch {epoch_no}")
                    print(f"Best validation loss: {best_valid_loss:.6f}")
                    break
        
        # Nếu không có validation loader, chỉ lưu model cuối cùng
        elif valid_loader is None:
            if foldername != "":
                torch.save(model.state_dict(), output_path)

    # Lưu model cuối cùng nếu không có early stopping
    if foldername != "" and valid_loader is None:
        torch.save(model.state_dict(), output_path)


def quantile_loss(target, forecast, q: float, eval_points) -> float:
    return 2 * torch.sum(
        torch.abs((forecast - target) * eval_points * ((target <= forecast) * 1.0 - q))
    )


def calc_denominator(target, eval_points):
    return torch.sum(torch.abs(target * eval_points))


def calc_quantile_CRPS(target, forecast, eval_points, mean_scaler, scaler):

    target = target * scaler + mean_scaler
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = []
        for j in range(len(forecast)):
            q_pred.append(torch.quantile(forecast[j : j + 1], quantiles[i], dim=1))
        q_pred = torch.cat(q_pred, 0)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def calc_quantile_CRPS_sum(target, forecast, eval_points, mean_scaler, scaler):

    eval_points = eval_points.mean(-1)
    target = target * scaler + mean_scaler
    target = target.sum(-1)
    forecast = forecast * scaler + mean_scaler

    quantiles = np.arange(0.05, 1.0, 0.05)
    denom = calc_denominator(target, eval_points)
    CRPS = 0
    for i in range(len(quantiles)):
        q_pred = torch.quantile(forecast.sum(-1),quantiles[i],dim=1)
        q_loss = quantile_loss(target, q_pred, quantiles[i], eval_points)
        CRPS += q_loss / denom
    return CRPS.item() / len(quantiles)

def evaluate(model,stock,seq,pred, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername="", top_k = 1, step_size = 1, seed = 2024):
    with torch.no_grad():
        model.eval()
        mse_total = 0
        mae_total = 0
        evalpoints_total = 0
        var_total = 0.0
        x = np.linspace(0, 100, 100)
        mse_list = np.zeros(100)
        all_target = []
        all_observed_point = []
        all_observed_time = []
        all_evalpoint = []
        all_generated_samples = []
        with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
            for batch_no, test_batch in enumerate(it, start=1):
                output = model.evaluate(test_batch, nsample)

                samples, c_target, eval_points, observed_points, observed_time = output
                samples = samples.permute(0, 1, 3, 2)  # (B, nsample, L, K)
                
                c_target = c_target.permute(0, 2, 1)     # (B, L, K)
                
                eval_points = eval_points.permute(0, 2, 1)
                observed_points = observed_points.permute(0, 2, 1)

                samples_median = samples.median(dim=1)
                samples_var = samples.var(dim=1)
                var_total += (samples_var * eval_points).sum().item()
                all_target.append(c_target)
                all_evalpoint.append(eval_points)
                all_observed_point.append(observed_points)
                all_observed_time.append(observed_time)
                all_generated_samples.append(samples)

                mse_current = (((samples_median.values - c_target) * eval_points) ** 2) * (scaler ** 2)
                mae_current = (torch.abs((samples_median.values - c_target) * eval_points)) * scaler
                if batch_no < 100:
                    mse_list[batch_no-1] = mse_current.sum().item() / eval_points.sum().item()
                mse_total += mse_current.sum().item()
                mae_total += mae_current.sum().item()
                evalpoints_total += eval_points.sum().item()

                it.set_postfix(
                    ordered_dict={
                        "rmse_total": np.sqrt(mse_total / evalpoints_total),
                        "mae_total": mae_total / evalpoints_total,
                        "batch_no": batch_no,
                    },
                    refresh=True,
                )
            fig, ax = plt.subplots()
            ax.plot(x, mse_list, color='#1D2B53')
            #plt.savefig('moti1.pdf')
            #plt.show()
            with open(foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb") as f:
                all_target = torch.cat(all_target, dim=0)
                all_evalpoint = torch.cat(all_evalpoint, dim=0)
                all_observed_point = torch.cat(all_observed_point, dim=0)
                all_observed_time = torch.cat(all_observed_time, dim=0)
                all_generated_samples = torch.cat(all_generated_samples, dim=0)
                pickle.dump(
                    [all_generated_samples, all_target, all_evalpoint, all_observed_point, all_observed_time, scaler, mean_scaler],
                    f,
                )

            CRPS = calc_quantile_CRPS(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)
            # CRPS_sum = calc_quantile_CRPS_sum(all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler)

            # with open(foldername + "/result_nsample" + str(nsample) + ".pk", "wb") as f:
            #     pickle.dump([np.sqrt(mse_total / evalpoints_total), mae_total / evalpoints_total, CRPS], f)
            #     print("RMSE:", np.sqrt(mse_total / evalpoints_total))
            #     print("MAE:", mae_total / evalpoints_total)
            #     print("CRPS:", CRPS)
            #     print("CRPS_sum:", CRPS_sum)
                
    # Thêm return các giá trị cần thiết cho việc vẽ đồ thị
    #return all_generated_samples, all_target, all_evalpoint, all_observed_point, all_observed_time
    MSE = mse_total / evalpoints_total
    MAE = mae_total / evalpoints_total
    variance = var_total / evalpoints_total
    print(f"MSE: {MSE}")
    print(evalpoints_total)
    print(mse_total)
    import pandas as pd
    import os
    
    # filename_excel = "Nor_nor_all_reports.xlsx"
    filename_csv = f"reports_{stock}.csv"
    # filename_excel = "Industry_reports.xlsx"
    # filename_csv = "Industry_reports.csv"
    
    data = {
        "dataset": "only",
        "stock": [stock],
        "seq" : [seq],
        "pred": [pred],
        "seed": [seed],
        "step_size": [step_size],
        "top_k": [top_k],
        "MSE": [MSE],
        "MAE": [MAE],
        "CRPS": [CRPS],
        "variance": [variance],
        "learning rate": 3.0e-4,
        "epochs": 100,
        "batch_size": 8,
        "reference": True
    }
    
    df = pd.DataFrame(data)
    ## Excel file
    # if os.path.exists(filename_excel):
    #     with pd.ExcelWriter(filename_excel, engine='openpyxl', mode='a', if_sheet_exists='overlay') as writer:
    #         reader = pd.read_excel(filename_excel)
    #         df.to_excel(writer, index=False, header=False, startrow=len(reader)+1)
    # else:
    #     df.to_excel(filename_excel, index=False)

    ## CSV file
    if os.path.exists(filename_csv):
        df.to_csv(filename_csv, mode='a', index=False, header=False)
    else:
        df.to_csv(filename_csv, index=False)




    
    return mse_total / evalpoints_total, mae_total / evalpoints_total

# Experiment n k 



## Hàm cũ
# def evaluate(model, test_loader, nsample=100, scaler=1, mean_scaler=0, foldername=""):
#     with torch.no_grad():
#         model.eval()
#         mse_total = 0
#         mae_total = 0
#         evalpoints_total = 0
#         x=np.linspace(0,100,100)
#         mse_list=np.zeros(100)
#         all_target = []
#         all_observed_point = []
#         all_observed_time = []
#         all_evalpoint = []
#         all_generated_samples = []
#         with tqdm(test_loader, mininterval=5.0, maxinterval=50.0) as it:
#             for batch_no, test_batch in enumerate(it, start=1):
#                 output = model.evaluate(test_batch, nsample)

#                 samples, c_target, eval_points, observed_points, observed_time = output
#                 samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
#                 c_target = c_target.permute(0, 2, 1)  # (B,L,K)
#                 eval_points = eval_points.permute(0, 2, 1)
#                 observed_points = observed_points.permute(0, 2, 1)

#                 samples_median = samples.median(dim=1)
#                 all_target.append(c_target)
#                 all_evalpoint.append(eval_points)
#                 all_observed_point.append(observed_points)
#                 all_observed_time.append(observed_time)
#                 all_generated_samples.append(samples)
                
                
#                 mse_current = (
#                     ((samples_median.values - c_target) * eval_points) ** 2
#                 ) * (scaler ** 2)
#                 mae_current = (
#                     torch.abs((samples_median.values - c_target) * eval_points) 
#                 ) * scaler
#                 if batch_no<100:
#                     mse_list[batch_no-1]=mse_current.sum().item()/eval_points.sum().item()
#                 mse_total += mse_current.sum().item()
#                 mae_total += mae_current.sum().item()
#                 evalpoints_total += eval_points.sum().item()

#                 it.set_postfix(
#                     ordered_dict={
#                         "rmse_total": np.sqrt(mse_total / evalpoints_total),
#                         "mae_total": mae_total / evalpoints_total,
#                         "batch_no": batch_no,
#                     },
#                     refresh=True,
#                 )
#             fig,ax = plt.subplots()
#             ax.plot(x,mse_list,color = '#1D2B53')
#             plt.savefig('moti1.pdf')
#             plt.show()
#             with open(
#                 foldername + "/generated_outputs_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 all_target = torch.cat(all_target, dim=0)
#                 all_evalpoint = torch.cat(all_evalpoint, dim=0)
#                 all_observed_point = torch.cat(all_observed_point, dim=0)
#                 all_observed_time = torch.cat(all_observed_time, dim=0)
#                 all_generated_samples = torch.cat(all_generated_samples, dim=0)

#                 pickle.dump(
#                     [
#                         all_generated_samples,
#                         all_target,
#                         all_evalpoint,
#                         all_observed_point,
#                         all_observed_time,
#                         scaler,
#                         mean_scaler,
#                     ],
#                     f,
#                 )

#             CRPS = calc_quantile_CRPS(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )
#             CRPS_sum = calc_quantile_CRPS_sum(
#                 all_target, all_generated_samples, all_evalpoint, mean_scaler, scaler
#             )

#             with open(
#                 foldername + "/result_nsample" + str(nsample) + ".pk", "wb"
#             ) as f:
#                 pickle.dump(
#                     [
#                         np.sqrt(mse_total / evalpoints_total),
#                         mae_total / evalpoints_total,
#                         CRPS,
#                     ],
#                     f,
#                 )
#                 print("RMSE:", np.sqrt(mse_total / evalpoints_total))
#                 print("MAE:", mae_total / evalpoints_total)
#                 print("CRPS:", CRPS)
#                 print("CRPS_sum:", CRPS_sum)
