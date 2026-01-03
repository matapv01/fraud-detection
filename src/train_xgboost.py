import xgboost as xgb
import numpy as np
import joblib
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt

def train_xgboost(
    df_train,
    df_test,
    target_col='Class',
    save_model_path='checkpoint/xgb_model.pkl',
    save_plot_prefix='xgb'
):
    # =========================
    # 1️⃣ Prepare data
    # =========================
    X_train = df_train.drop(columns=[target_col]).values
    y_train = df_train[target_col].values
    X_test  = df_test.drop(columns=[target_col]).values
    y_test  = df_test[target_col].values

    # =========================
    # 2️⃣ Compute class weight
    # =========================
    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # =========================
    # 3️⃣ Model
    # =========================
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )

    # =========================
    # 4️⃣ Train
    # =========================
    model.fit(X_train, y_train)

    joblib.dump(model, save_model_path)
    print(f"XGBoost model saved to {save_model_path}")

    # =========================
    # 5️⃣ Evaluate
    # =========================
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:,1]

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC AUC: {auc:.4f}")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(6,6))
    plt.plot(fpr, tpr, label=f'XGB (AUC={auc:.4f})')
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend()
    plt.grid()
    plt.savefig(f'{save_plot_prefix}_roc.png')
    plt.show()

    return model, auc
