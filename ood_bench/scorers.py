import torch
import torch.nn as nn
from typing import Optional

def compute_msp(logits: torch.Tensor) -> torch.Tensor:
    """
    Computes Maximum Softmax Probability (MSP).
    Returns values in [0, 1]. Higher indicates higher in-distribution confidence.
    """
    probs = torch.softmax(logits, dim=-1)
    msp_scores, _ = torch.max(probs, dim=-1)
    return msp_scores


def compute_energy(logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """
    Computes Helmholtz Free Energy using LogSumExp for numerical stability:
    S_energy(x) = T * log(sum(exp(logits / T)))
    Higher score indicates in-distribution; lower score indicates out-of-distribution.
    """
    scaled_logits = logits / temperature
    # PyTorch logsumexp avoids intermediate float overflow
    energy_scores = temperature * torch.logsumexp(scaled_logits, dim=-1)
    return energy_scores


class MahalanobisScorer:
    """
    Computes minimum Mahalanobis distance from penultimate latent activations
    to class-conditional Gaussian centroids.
    """
    def __init__(self, num_classes: int, feature_dim: int):
        self.num_classes = num_classes
        self.feature_dim = feature_dim
        self.centroids: Optional[torch.Tensor] = None
        self.precision: Optional[torch.Tensor] = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor):
        """Fit empirical class centroids and shared tied covariance matrix."""
        self.centroids = torch.zeros(self.num_classes, self.feature_dim)
        shared_cov = torch.zeros(self.feature_dim, self.feature_dim)

        for c in range(self.num_classes):
            class_feats = features[labels == c]
            if len(class_feats) == 0:
                continue
            mean_c = class_feats.mean(dim=0)
            self.centroids[c] = mean_c
            diff = class_feats - mean_c
            shared_cov += diff.t().mm(diff)

        shared_cov /= len(features)
        # Add small ridge regularization to guarantee positive-definite inversion
        reg_cov = shared_cov + 1e-4 * torch.eye(self.feature_dim)
        self.precision = torch.linalg.inv(reg_cov)

    def score(self, features: torch.Tensor) -> torch.Tensor:
        """
        Returns the negative minimum Mahalanobis distance:
        Higher score = closer to an ID centroid (In-Distribution).
        Lower score = far from all centroids (Out-of-Distribution).
        """
        if self.centroids is None or self.precision is None:
            raise RuntimeError("MahalanobisScorer must be fit on training data before scoring.")

        batch_size = features.shape[0]
        min_dists = torch.full((batch_size,), float('inf'))

        for c in range(self.num_classes):
            diff = features - self.centroids[c]
            # dist^2 = (z - mu)^T * Sigma^{-1} * (z - mu)
            dist_sq = torch.sum(diff.mm(self.precision) * diff, dim=-1)
            dist = torch.sqrt(torch.clamp(dist_sq, min=0.0))
            min_dists = torch.minimum(min_dists, dist)

        # Negative distance so that higher values correlate with In-Distribution
        return -min_dists