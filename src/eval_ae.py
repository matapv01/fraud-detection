from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_auc_score, 
    roc_curve
)
import numpy as np
import torch
import matplotlib.pyplot as plt


def find_best_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr


def evaluate_ae(
    model_class,       
    model_path,        
    recon_error_path,  
    test_df,           
    target_col='Class',
    latent_dim=8
):
    # --------------------------
    # 1️⃣ Load reconstruction error
    # --------------------------
    recon_error_test = np.load(recon_error_path)
    y_test = test_df[target_col].values

    # --------------------------
    # 2️⃣ Load AE model
    # --------------------------
    input_dim = test_df.drop(columns=[target_col]).shape[1]
    model = model_class(input_dim=input_dim, latent_dim=latent_dim)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    print("Model loaded:", model_path)

    # --------------------------
    # 3️⃣ Threshold tối ưu (ROC Youden’s J)
    # --------------------------
    threshold, fpr, tpr = find_best_threshold(y_test, recon_error_test)
    print("Optimal anomaly threshold:", threshold)

    # --------------------------
    # 4️⃣ Predict
    # --------------------------
    anomaly_pred = (recon_error_test > threshold).astype(int)

    # --------------------------
    # 5️⃣ Metrics
    # --------------------------
    print("\nClassification Report:")
    print(classification_report(y_test, anomaly_pred))

    cm = confusion_matrix(y_test, anomaly_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix (AE)")
    plt.show()

    # ROC AUC
    auc_score = roc_auc_score(y_test, recon_error_test)
    print(f"ROC AUC: {auc_score:.4f}")

    # ROC Curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc_score:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (AE)")
    plt.legend()
    plt.grid()
    plt.show()

    return model, anomaly_pred, threshold, auc_score, recon_error_test
