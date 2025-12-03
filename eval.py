import pandas as pd
from sklearn.model_selection import train_test_split
import joblib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing import scale_amount_time_train_test, remove_extreme_outliers, scale_test_with_saved_scaler

def evaluate_saved_model(df, model_path, target_col='Class', test_size=0.1, random_state=42,
                         save_plot_prefix='checkpoint/lr_test'):
    """
    Load mô hình đã train, đánh giá trên test (10%), in classification metrics,
    in confusion matrix dạng text, vẽ confusion matrix và ROC curve.
    """
    # Chia test data
    X = df.drop(columns=[target_col])
    y = df[target_col]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=None
    )

    # Scale test set
    X_test = scale_test_with_saved_scaler(X_test)

    # Load model
    model = joblib.load(model_path)
    print(f"Model loaded from: {model_path}")
    
    # Predict
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]
    
    # Classification report
    print("\nClassification Report (Test set):\n", classification_report(y_test, y_pred))
    auc_score = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC (Test set): {auc_score:.4f}")
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # 👉 IN RA CONFUSION MATRIX DẠNG TEXT
    print("\nConfusion Matrix (text format):")
    print(cm)

    # Vẽ Confusion Matrix heatmap
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal','Fraud'], yticklabels=['Normal','Fraud'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig(f'{save_plot_prefix}_confusion_matrix.png')
    plt.show()
    
    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'Model (AUC = {auc_score:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve (Test set)')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_plot_prefix}_roc.png')
    plt.show()
    
    # Return metrics
    test_metrics = {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'roc_auc': auc_score,
        'confusion_matrix': cm
    }
    
    return test_metrics

# Example usage:
df = pd.read_csv("/kaggle/input/creditcardfraud/creditcard.csv")
evaluate_saved_model(df, model_path='lr_model.pkl')
