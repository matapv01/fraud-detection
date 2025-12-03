from imblearn.over_sampling import SMOTE
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, 
    roc_auc_score, 
    roc_curve, 
    confusion_matrix, 
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt


def train_lr_full_train(
    df_train, 
    df_test, 
    target_col='Class', 
    save_plot_prefix='checkpoint/lr', 
    save_model_path='checkpoint/lr_model.pkl',
    max_iter=200,
    class_weights='balanced',
    use_smote=False
):
    # Tách features và label
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Áp dụng SMOTE nếu cần
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train, y_train = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE, class distribution:\n{y_train.value_counts()}")

    # Train Logistic Regression
    lr = LogisticRegression(
        class_weight=class_weights,
        random_state=42,
        max_iter=max_iter
    )
    lr.fit(X_train, y_train)

    # Lưu model
    joblib.dump(lr, save_model_path)
    print(f"\nModel saved to: {save_model_path}")

    # Dự đoán
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:,1]

    # Classification report
    print("\nClassification Report (Test set):\n", classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC (Test set): {auc_score:.4f}")

    # *** Confusion Matrix ***
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    # Plot Confusion Matrix
    plt.figure(figsize=(5,5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix (Test set)')
    plt.grid(False)
    plt.savefig(f'{save_plot_prefix}_confusion_matrix.png')
    plt.show()

    # Vẽ ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'LR (AUC = {auc_score:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test set)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_plot_prefix}_roc.png')
    plt.show()

    return lr, {
        'y_test': y_test, 
        'y_pred': y_pred, 
        'y_proba': y_proba, 
        'roc_auc': auc_score,
        'confusion_matrix': cm
    }
