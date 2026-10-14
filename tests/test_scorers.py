import torch
import numpy as np
import pytest
from ood_bench.scorers import compute_msp, compute_energy, MahalanobisScorer
from ood_bench.metrics import compute_auroc, compute_fpr95

def test_msp_bounds():
    logits = torch.randn(10, 5)
    scores = compute_msp(logits)
    assert scores.shape == (10,)
    assert (scores >= 0.0).all() and (scores <= 1.0).all()

def test_energy_higher_for_confident_in_distribution():
    # ID sample with strong positive logits
    id_logits = torch.tensor([[10.0, 1.0, 0.5]])
    # OOD sample with suppressed/uniform logits
    ood_logits = torch.tensor([[0.2, 0.1, 0.3]])

    id_energy = compute_energy(id_logits)
    ood_energy = compute_energy(ood_logits)

    # Higher score = more in-distribution
    assert id_energy.item() > ood_energy.item()

def test_mahalanobis_closer_centroid():
    scorer = MahalanobisScorer(num_classes=2, feature_dim=2)
    train_feats = torch.tensor([[1.0, 1.0], [1.1, 0.9], [-1.0, -1.0], [-0.9, -1.1]])
    train_labels = torch.tensor([0, 0, 1, 1])
    scorer.fit(train_feats, train_labels)

    near_id = torch.tensor([[1.0, 1.0]])
    far_ood = torch.tensor([[20.0, 20.0]])

    score_id = scorer.score(near_id).item()
    score_ood = scorer.score(far_ood).item()

    assert score_id > score_ood

def test_metrics_perfect_separation():
    id_scores = np.array([0.9, 0.85, 0.95, 0.88, 0.92])
    ood_scores = np.array([0.1, 0.15, 0.05, 0.2, 0.12])

    auroc = compute_auroc(id_scores, ood_scores)
    fpr95 = compute_fpr95(id_scores, ood_scores)

    assert auroc == 1.0
    assert fpr95 == 0.0