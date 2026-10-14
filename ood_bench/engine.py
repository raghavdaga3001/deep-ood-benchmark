import torch
import torch.nn as nn
import numpy as np
from tabulate import tabulate
from ood_bench.scorers import compute_msp, compute_energy, MahalanobisScorer
from ood_bench.metrics import evaluate_scorer

def run_benchmark():
    torch.manual_seed(42)
    np.random.seed(42)

    num_classes = 10
    feature_dim = 64
    n_train = 1000
    n_test = 500

    print("[*] Generating controlled representation manifolds...")

    # 1. Class Centroids in Latent Space
    centroids = torch.randn(num_classes, feature_dim) * 3.0

    # 2. Train Features (ID)
    train_labels = torch.randint(0, num_classes, (n_train,))
    train_features = centroids[train_labels] + torch.randn(n_train, feature_dim) * 0.8

    # 3. Test Features (In-Distribution)
    test_id_labels = torch.randint(0, num_classes, (n_test,))
    test_id_features = centroids[test_id_labels] + torch.randn(n_test, feature_dim) * 0.8

    # 4. Out-of-Distribution (OOD) Features: Generated in latent feature voids
    # Located far from any class centroid, but with directional variance
    ood_features = torch.randn(n_test, feature_dim) * 5.0

    # 5. Simulated Linear Classifier Head: W * z + b
    classifier = nn.Linear(feature_dim, num_classes)
    with torch.no_grad():
        # Initialize linear layer to project centroids to distinct positive logits
        classifier.weight.copy_(centroids)
        classifier.bias.zero_()

    # 6. Forward passes
    with torch.no_grad():
        id_logits = classifier(test_id_features)
        ood_logits = classifier(ood_features)

        # Artificially inject softmax distortion: scale one random logit on OOD samples
        # This precisely simulates how high-magnitude logits create false softmax confidence
        boost_idx = torch.randint(0, num_classes, (n_test,))
        for i in range(n_test):
            ood_logits[i, boost_idx[i]] += 4.5

    # 7. Evaluate Baseline: Maximum Softmax Probability (MSP)
    msp_id = compute_msp(id_logits).numpy()
    msp_ood = compute_msp(ood_logits).numpy()
    msp_auroc, msp_fpr95 = evaluate_scorer(msp_id, msp_ood)

    # 8. Evaluate Helmholtz Free Energy (T = 1.0)
    energy_id = compute_energy(id_logits, temperature=1.0).numpy()
    energy_ood = compute_energy(ood_logits, temperature=1.0).numpy()
    energy_auroc, energy_fpr95 = evaluate_scorer(energy_id, energy_ood)

    # 9. Evaluate Mahalanobis Distance
    maha_scorer = MahalanobisScorer(num_classes=num_classes, feature_dim=feature_dim)
    maha_scorer.fit(train_features, train_labels)
    maha_id = maha_scorer.score(test_id_features).numpy()
    maha_ood = maha_scorer.score(ood_features).numpy()
    maha_auroc, maha_fpr95 = evaluate_scorer(maha_id, maha_ood)

    results = [
        ["Maximum Softmax Probability (MSP)", f"{msp_auroc * 100:.2f}%", f"{msp_fpr95 * 100:.2f}%"],
        ["Helmholtz Free Energy (T=1.0)", f"{energy_auroc * 100:.2f}%", f"{energy_fpr95 * 100:.2f}%"],
        ["Mahalanobis Latent Distance", f"{maha_auroc * 100:.2f}%", f"{maha_fpr95 * 100:.2f}%"]
    ]

    print("\n" + "=" * 65)
    print("OOD DETECTION BENCHMARK RESULTS (ID vs. Feature Void Anomaly)")
    print("=" * 65)
    print(tabulate(results, headers=["Scoring Method", "AUROC (↑)", "FPR95 (↓)"], tablefmt="github"))
    print("=" * 65 + "\n")
    return results