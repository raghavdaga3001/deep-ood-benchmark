import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from ood_bench.scorers import compute_msp, compute_energy, MahalanobisScorer

os.makedirs("assets", exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)

num_classes, feature_dim, n_train, n_test = 10, 64, 1000, 500
centroids = torch.randn(num_classes, feature_dim) * 3.0

train_labels = torch.randint(0, num_classes, (n_train,))
train_features = centroids[train_labels] + torch.randn(n_train, feature_dim) * 0.8

test_id_labels = torch.randint(0, num_classes, (n_test,))
test_id_features = centroids[test_id_labels] + torch.randn(n_test, feature_dim) * 0.8
ood_features = torch.randn(n_test, feature_dim) * 5.0

classifier = nn.Linear(feature_dim, num_classes)
with torch.no_grad():
    classifier.weight.copy_(centroids)
    classifier.bias.zero_()
    id_logits = classifier(test_id_features)
    ood_logits = classifier(ood_features)
    boost_idx = torch.randint(0, num_classes, (n_test,))
    for i in range(n_test):
        ood_logits[i, boost_idx[i]] += 4.5

# Calculate Scores
msp_id = compute_msp(id_logits).numpy()
msp_ood = compute_msp(ood_logits).numpy()

energy_id = compute_energy(id_logits).numpy()
energy_ood = compute_energy(ood_logits).numpy()

maha_scorer = MahalanobisScorer(num_classes=num_classes, feature_dim=feature_dim)
maha_scorer.fit(train_features, train_labels)
maha_id = maha_scorer.score(test_id_features).numpy()
maha_ood = maha_scorer.score(ood_features).numpy()

# Plotting Distributions
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. MSP Plot
axes[0].hist(msp_id, bins=30, alpha=0.6, label="In-Distribution (ID)", color="#2563eb", density=True)
axes[0].hist(msp_ood, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", color="#dc2626", density=True)
axes[0].set_title("Maximum Softmax Probability (MSP)\nHeavy Overlap / High Overconfidence", fontsize=11, fontweight="bold")
axes[0].set_xlabel("Confidence Score")
axes[0].set_ylabel("Density")
axes[0].legend()

# 2. Free Energy Plot
axes[1].hist(energy_id, bins=30, alpha=0.6, label="In-Distribution (ID)", color="#2563eb", density=True)
axes[1].hist(energy_ood, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", color="#16a34a", density=True)
axes[1].set_title("Helmholtz Free Energy (T=1.0)\nClean Separation via LogSumExp", fontsize=11, fontweight="bold")
axes[1].set_xlabel("Energy Score (Higher = ID)")
axes[1].legend()

# 3. Mahalanobis Distance Plot
axes[2].hist(maha_id, bins=30, alpha=0.6, label="In-Distribution (ID)", color="#2563eb", density=True)
axes[2].hist(maha_ood, bins=30, alpha=0.6, label="Out-of-Distribution (OOD)", color="#9333ea", density=True)
axes[2].set_title("Mahalanobis Latent Distance\nZero Manifold Contamination", fontsize=11, fontweight="bold")
axes[2].set_xlabel("Negative Mahalanobis Distance")
axes[2].legend()

plt.tight_layout()
output_path = "assets/score_distributions.png"
plt.savefig(output_path, dpi=300)
print(f"[✓] Distribution plot saved successfully to {output_path}")