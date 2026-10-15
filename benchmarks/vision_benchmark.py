import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
import numpy as np
from tabulate import tabulate

from ood_bench.scorers import compute_msp, compute_energy, MahalanobisScorer
from ood_bench.metrics import evaluate_scorer

# Limit CPU threads to keep machine cool
torch.set_num_threads(2)

DEVICE = torch.device("cpu")
SAMPLE_SIZE = 500
BATCH_SIZE = 64

def get_cifar_transforms():
    return transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616]
        )
    ])

def load_data():
    """Loads lightweight test slices for ID, Near-OOD, and Far-OOD."""
    transform = get_cifar_transforms()

    # 1. In-Distribution (CIFAR-10 Test)
    cifar10_raw = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
    cifar10_loader = DataLoader(Subset(cifar10_raw, range(SAMPLE_SIZE)), batch_size=BATCH_SIZE, shuffle=False)

    # 1b. Small Train Slice for Fitting Mahalanobis Covariance
    cifar10_train = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(Subset(cifar10_train, range(1000)), batch_size=BATCH_SIZE, shuffle=False)

    # 2. Near-OOD (CIFAR-100 Test)
    cifar100_raw = datasets.CIFAR100(root="./data", train=False, download=True, transform=transform)
    cifar100_loader = DataLoader(Subset(cifar100_raw, range(SAMPLE_SIZE)), batch_size=BATCH_SIZE, shuffle=False)

    # 3. Far-OOD (SVHN Test)
    svhn_transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
    ])
    svhn_raw = datasets.SVHN(root="./data", split="test", download=True, transform=svhn_transform)
    svhn_loader = DataLoader(Subset(svhn_raw, range(SAMPLE_SIZE)), batch_size=BATCH_SIZE, shuffle=False)

    return train_loader, cifar10_loader, cifar100_loader, svhn_loader

def extract_features_and_logits(model, dataloader):
    """Extracts penultimate representations and classification logits."""
    features_list = []
    logits_list = []
    labels_list = []

    # Penultimate hook
    latent_cache = []
    def hook_fn(module, input, output):
        latent_cache.append(torch.flatten(output, 1))

    hook = model.avgpool.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (tuple, list)):
                x, y = batch[0].to(DEVICE), batch[1].to(DEVICE)
            else:
                x, y = batch.to(DEVICE), torch.zeros(len(batch))

            logits = model(x)
            logits_list.append(logits)
            features_list.append(latent_cache[-1])
            labels_list.append(y)

    hook.remove()
    return (
        torch.cat(features_list, dim=0),
        torch.cat(logits_list, dim=0),
        torch.cat(labels_list, dim=0)
    )

def run_vision_benchmark():
    print("[*] Loading vision benchmarks (CIFAR-10 / CIFAR-100 / SVHN)...")
    train_loader, id_loader, near_ood_loader, far_ood_loader = load_data()

    print("[*] Loading pretrained ResNet-18 backbone...")
    # ResNet-18 pretrained backbone adapted for 10 classes
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.eval()

    print("[*] Extracting latent features and logits...")
    train_feats, _, train_labels = extract_features_and_logits(model, train_loader)
    id_feats, id_logits, _ = extract_features_and_logits(model, id_loader)
    near_feats, near_logits, _ = extract_features_and_logits(model, near_ood_loader)
    far_feats, far_logits, _ = extract_features_and_logits(model, far_ood_loader)

    print("[*] Fitting class centroids & covariance for Mahalanobis Scorer...")
    maha = MahalanobisScorer(num_classes=10, feature_dim=train_feats.shape[1])
    maha.fit(train_feats, train_labels)

    # In-Distribution Scores
    msp_id = compute_msp(id_logits).numpy()
    energy_id = compute_energy(id_logits, temperature=1.0).numpy()
    maha_id = maha.score(id_feats).numpy()

    # Far-OOD Scores (SVHN)
    msp_far = compute_msp(far_logits).numpy()
    energy_far = compute_energy(far_logits, temperature=1.0).numpy()
    maha_far = maha.score(far_feats).numpy()

    # Near-OOD Scores (CIFAR-100)
    msp_near = compute_msp(near_logits).numpy()
    energy_near = compute_energy(near_logits, temperature=1.0).numpy()
    maha_near = maha.score(near_feats).numpy()

    # Metric Evaluations
    results = [
        ["Far-OOD: SVHN", "MSP (Baseline)", f"{evaluate_scorer(msp_id, msp_far)[0] * 100:.2f}%", f"{evaluate_scorer(msp_id, msp_far)[1] * 100:.2f}%"],
        ["Far-OOD: SVHN", "Helmholtz Energy", f"{evaluate_scorer(energy_id, energy_far)[0] * 100:.2f}%", f"{evaluate_scorer(energy_id, energy_far)[1] * 100:.2f}%"],
        ["Far-OOD: SVHN", "Mahalanobis Latent", f"{evaluate_scorer(maha_id, maha_far)[0] * 100:.2f}%", f"{evaluate_scorer(maha_id, maha_far)[1] * 100:.2f}%"],
        ["------------------", "------------------", "----------", "----------"],
        ["Near-OOD: CIFAR-100", "MSP (Baseline)", f"{evaluate_scorer(msp_id, msp_near)[0] * 100:.2f}%", f"{evaluate_scorer(msp_id, msp_near)[1] * 100:.2f}%"],
        ["Near-OOD: CIFAR-100", "Helmholtz Energy", f"{evaluate_scorer(energy_id, energy_near)[0] * 100:.2f}%", f"{evaluate_scorer(energy_id, energy_near)[1] * 100:.2f}%"],
        ["Near-OOD: CIFAR-100", "Mahalanobis Latent", f"{evaluate_scorer(maha_id, maha_near)[0] * 100:.2f}%", f"{evaluate_scorer(maha_id, maha_near)[1] * 100:.2f}%"],
    ]

    print("\n" + "=" * 75)
    print("VISION OOD BENCHMARK: CIFAR-10 (ID) vs. SVHN (Far) & CIFAR-100 (Near)")
    print("=" * 75)
    print(tabulate(results, headers=["Dataset Pair", "Scorer", "AUROC (↑)", "FPR95 (↓)"], tablefmt="github"))
    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_vision_benchmark()