import torch
import numpy as np
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_auc_score, 
    roc_curve
)
import matplotlib.pyplot as plt

def evaluate_ae(
    model_class,       # class Autoencoder
    model_path,        # path .pt
    recon_error_path,  # path .npy
    test_df,           # DataFrame test, phải có cột target_col
    target_col='Class',
    latent_dim=8,
    threshold_method='mean+2std'  # cách tính threshold
):
    # -----------------------------
    # 1️⃣ Load reconstruction error
    # -----------------------------
    recon_error_test = np.load(recon_error_path)
    print("Loaded reconstruction error:", recon_error_test.shape)

    # -----------------------------
    # 2️⃣ Load AE model
    # -----------------------------
    input_dim = test_df.drop(columns=[target_col]).shape[1]
    model = model_class(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print(f"Model loaded from {model_path}")

    # -----------------------------
    # 3️⃣ Tính threshold
    # -----------------------------
    if threshold_method == 'mean+3std':
        # Nếu bạn không có recon_error_train, dùng reconstruction error test để demo
        threshold = recon_error_test.mean() + 3*recon_error_test.std()
    elif threshold_method == 'mean+2std':
        threshold = recon_error_test.mean() + 2*recon_error_test.std()
    else:
        raise ValueError("Currently only support 'mean+3std' and 'mean+2std' thresholds")
    print("Threshold for anomaly:", threshold)

    # -----------------------------
    # 4️⃣ Dự đoán anomaly
    # -----------------------------
    y_test = test_df[target_col].values
    anomaly_pred = (recon_error_test > threshold).astype(int)

    # -----------------------------
    # 5️⃣ Metrics
    # -----------------------------
    print("\nClassification Report:")
    print(classification_report(y_test, anomaly_pred))

    # Confusion Matrix
    cm = confusion_matrix(y_test, anomaly_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title("Confusion Matrix (Test set)")
    plt.show()

    # ROC AUC
    auc_score = roc_auc_score(y_test, recon_error_test)
    print(f"ROC AUC (Test set): {auc_score:.4f}")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, recon_error_test)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'AE (AUC = {auc_score:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test set)')
    plt.legend()
    plt.grid()
    plt.show()

    return model, anomaly_pred, threshold, auc_score, recon_error_test
