import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from typing import Tuple

def compute_auroc(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Computes Area Under the ROC Curve (AUROC).
    Assigns Label 1 to In-Distribution (ID) and Label 0 to Out-of-Distribution (OOD).
    AUROC = 1.0 represents perfect separation.
    """
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])
    return float(roc_auc_score(y_true, y_scores))


def compute_fpr95(id_scores: np.ndarray, ood_scores: np.ndarray) -> float:
    """
    Computes False Positive Rate at 95% True Positive Rate (FPR95).
    Measures OOD misclassification rate when 95% of ID samples are correctly preserved.
    Lower is better (0.0 represents zero false alarms on anomalies).
    """
    y_true = np.concatenate([np.ones_like(id_scores), np.zeros_like(ood_scores)])
    y_scores = np.concatenate([id_scores, ood_scores])

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Locate the operating threshold where True Positive Rate >= 0.95
    idx = np.where(tpr >= 0.95)[0]
    if len(idx) == 0:
        return 1.0
    
    fpr_at_95 = fpr[idx[0]]
    return float(fpr_at_95)


def evaluate_scorer(id_scores: np.ndarray, ood_scores: np.ndarray) -> Tuple[float, float]:
    """Returns (AUROC, FPR95)."""
    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr95(id_scores, ood_scores)
    return auroc, fpr95