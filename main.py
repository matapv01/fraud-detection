import pandas as pd
import numpy as np


from src.data_analysis import data_analysis, distribution_plot, boxplot_outliers
from src.data_loader import data_loader
from src.preprocessing import remove_extreme_outliers, scale_amount_time_train_test
from src.train import train_lr_full_train
from src.train_ae import train_ae




df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")

train_df, test_df, new_df = data_loader(df)

# data_analysis(new_df, idx=0)
# distribution_plot(new_df, idx=0)

train_df_clean = remove_extreme_outliers(train_df)
# boxplot_outliers(train_df_clean, features=['V10', 'V12', 'V14'], save_name='boxplot_after_outlier_removal.png')

train_df_scaled, test_df_scaled, scaler_amount, scaler_time = scale_amount_time_train_test(train_df_clean, test_df)


# best_lr_model, test_metrics = train_lr_full_train(train_df_scaled, test_df_scaled, use_smote=True)





# Train AE
# ae_model, recon_error_test = train_ae(train_df_scaled, test_df_scaled, target_col='Class')

# eval AE
from src.train_ae import Autoencoder
from src.eval_ae import evaluate_ae

model_path = 'checkpoint/ae_model.pt'
recon_error_path = 'checkpoint/recon_error_test.npy'

ae_model, anomaly_pred, threshold, auc_score, recon_error_test = evaluate_ae(
    model_class=Autoencoder,
    model_path=model_path,
    recon_error_path=recon_error_path,
    test_df=test_df_scaled,
    target_col='Class',
    latent_dim=8
)

print(f"Detected anomalies: {anomaly_pred.sum()} / {len(anomaly_pred)}")
