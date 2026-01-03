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





# # Train AE
# ae_model, recon_error_test = train_ae(train_df_scaled, test_df_scaled, target_col='Class')

# # eval AE
# from src.train_ae import Autoencoder
# from src.eval_ae import evaluate_ae

# model_path = 'checkpoint/ae_model.pt'
# recon_error_path = 'checkpoint/recon_error_test.npy'

# ae_model, anomaly_pred, threshold, auc_score, recon_error_test = evaluate_ae(
#     model_class=Autoencoder,
#     model_path=model_path,
#     recon_error_path=recon_error_path,
#     test_df=test_df_scaled,
#     target_col='Class',
#     latent_dim=8
# )

# print(f"Detected anomalies: {anomaly_pred.sum()} / {len(anomaly_pred)}")




from src.train_xgboost import train_xgboost
from src.eval_xgboost import evaluate_xgboost
# # Train XGBoost
# xgb_model = train_xgboost(
#     df_train=train_df_scaled,
#     df_test=test_df_scaled,
#     target_col='Class',
#     save_model_path='checkpoint/xgb_model.pkl',
#     save_plot_prefix='xgb'
# )


# # Eval XGBoost
# evaluate_xgboost(
#     model_path='checkpoint/xgb_model.pkl',
#     test_df=test_df_scaled,
#     target_col='Class',
#     use_optimal_threshold=False
# )

from src.eval_topk import evaluate_topk
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
    roc_curve
)
import matplotlib.pyplot as plt
import numpy as np

# Load model
model = joblib.load("checkpoint/xgb_model.pkl")

X_test = test_df_scaled.drop(columns=['Class'])
y_test = test_df_scaled['Class'].values

# Score
y_score = model.predict_proba(X_test)[:, 1]

# Top-K evaluation
topk_result = evaluate_topk(
    y_true=y_test,
    scores=y_score,
    K=200
)

print("Recall@200:", topk_result["recall@K"])
print("Precision@200:", topk_result["precision@K"])
print("Implied threshold:", topk_result["threshold"])