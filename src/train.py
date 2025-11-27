from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import matplotlib.pyplot as plt

def train_lr_full_train(df_train, df_test, target_col='Class', save_plot_prefix='lr', C=1.0, max_iter=500):
    """
    Train Logistic Regression trên toàn bộ train set, đánh giá trực tiếp trên test set
    """
    # Tách features và label
    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]
    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    # Train Logistic Regression
    lr = LogisticRegression(
        C=C, # regularization strength
        penalty='l2', # L2 regularization -> avoid overfitting
        solver='liblinear',
        class_weight='balanced',
        random_state=42,
        max_iter=max_iter
    )
    lr.fit(X_train, y_train)

    # Dự đoán trên test set
    y_pred = lr.predict(X_test)
    y_proba = lr.predict_proba(X_test)[:,1]

    # Classification report
    print("\nClassification Report (Test set):\n", classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC (Test set): {auc_score:.4f}")

    # Vẽ ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'LR (AUC = {auc_score:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate') # tỉ lệ dự đoán âm nhưng bị mô hình dự đoán là dương
    plt.ylabel('True Positive Rate') # tỷ lệ dự đoán dương đúng
    plt.title('ROC Curve (Test set)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_plot_prefix}_roc.png')
    plt.show()

    return lr, {'y_test': y_test, 'y_pred': y_pred, 'y_proba': y_proba, 'roc_auc': auc_score}
