
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

def find_best_threshold(y_true, scores):
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], fpr, tpr

def evaluate_xgboost(
    model_path,
    test_df,
    target_col='Class',
    use_optimal_threshold=False
):
    # --------------------------
    # 1️⃣ Load model
    # --------------------------
    model = joblib.load(model_path)
    print("Model loaded:", model_path)

    X_test = test_df.drop(columns=[target_col]).values
    y_test = test_df[target_col].values

    # --------------------------
    # 2️⃣ Predict probabilities
    # --------------------------
    y_proba = model.predict_proba(X_test)[:,1]

    # --------------------------
    # 3️⃣ Threshold
    # --------------------------
    if use_optimal_threshold:
        threshold, fpr, tpr = find_best_threshold(y_test, y_proba)
        print("Optimal threshold (Youden J):", threshold)
    else:
        threshold = 0.5
        fpr, tpr, _ = roc_curve(y_test, y_proba)

    y_pred = (y_proba > threshold).astype(int)

    # --------------------------
    # 4️⃣ Metrics
    # --------------------------
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot(cmap="Blues")
    plt.title("Confusion Matrix (XGBoost)")
    plt.show()

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # ROC Curve
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0,1],[0,1],'--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC Curve (XGBoost)")
    plt.legend()
    plt.grid()
    plt.show()

    return y_pred, y_proba, threshold, auc
