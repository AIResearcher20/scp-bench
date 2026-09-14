import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)


def compute_metrics(y_true, y_pred, y_proba=None):
    """Compute 8 classification metrics."""
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred,
                                                  average="macro",
                                                  zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred,
                                            average="macro",
                                            zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred,
                                    average="macro",
                                    zero_division=0)),
        "f1_micro": float(f1_score(y_true, y_pred,
                                    average="micro",
                                    zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if y_proba is not None and len(np.unique(y_true)) > 1:
        try:
            if y_proba.shape[1] == 2:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                metrics["auroc"] = float(roc_auc_score(y_true, y_proba,
                                                        multi_class="ovr"))
        except Exception:
            pass

    return metrics
