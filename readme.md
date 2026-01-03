- LR
Classification Report (Test set):
               precision    recall  f1-score   support

           0       0.96      0.92      0.94        49
           1       0.92      0.96      0.94        49

    accuracy                           0.94        98
   macro avg       0.94      0.94      0.94        98
weighted avg       0.94      0.94      0.94        98



- AE 
Optimal anomaly threshold: 0.5585443

Classification Report:
              precision    recall  f1-score   support

           0       1.00      0.98      0.99     28432
           1       0.06      0.78      0.11        49

    accuracy                           0.98     28481
   macro avg       0.53      0.88      0.55     28481
weighted avg       1.00      0.98      0.99     28481

ROC AUC: 0.9258
Detected anomalies: 637 / 28481


- xgboost

--threshold 0.5
Classification Report:
              precision    recall  f1-score   support

           0       1.00      1.00      1.00     28432
           1       0.90      0.71      0.80        49

    accuracy                           1.00     28481
   macro avg       0.95      0.86      0.90     28481
weighted avg       1.00      1.00      1.00     28481

ROC AUC: 0.9640


--top200 (maximize Recall): Hệ thống chỉ cho phép flag 200 giao dịch/ngày để kiểm tra thủ công.
=== Top-200 Evaluation ===
              precision    recall  f1-score   support

           0       1.00      0.99      1.00     28432
           1       0.20      0.84      0.33        49

    accuracy                           0.99     28481
   macro avg       0.60      0.92      0.66     28481
weighted avg       1.00      0.99      1.00     28481

Recall@200: 0.8367346938775511
Precision@200: 0.205
Implied threshold: 0.009620405