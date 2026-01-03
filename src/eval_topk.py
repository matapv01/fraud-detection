import numpy as np
from sklearn.metrics import classification_report

def evaluate_topk(
    y_true,
    scores,
    K=200,
    verbose=True
):
    """
    Budget-based evaluation (Top-K)
    """
    # Sort theo score giảm dần
    order = np.argsort(scores)[::-1]
    y_true_sorted = y_true[order]

    # Predict Top-K
    y_pred = np.zeros_like(y_true)
    y_pred[order[:K]] = 1

    # Metrics
    if verbose:
        print(f"\n=== Top-{K} Evaluation ===")
        print(classification_report(y_true, y_pred))

    # Recall@K
    recall_at_k = y_true_sorted[:K].sum() / y_true.sum()

    # Precision@K
    precision_at_k = y_true_sorted[:K].sum() / K

    # Threshold chỉ để log
    threshold = scores[order[K-1]]

    return {
        "recall@K": recall_at_k,
        "precision@K": precision_at_k,
        "threshold": threshold,
        "y_pred": y_pred
    }
