"""
Federated learning prototype (experimental/research only).

Simulates N sites (e.g. hospitals) with partitioned data. Each site trains locally;
only model updates are aggregated (FedAvg). No raw patient data is shared.

Usage:
  python -m hemasight.ml.federated --data path/to/features.csv --label-col label --num-sites 3 --rounds 5
"""
import argparse
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn

from hemasight.ml.model_training import FEATURE_COLUMNS, load_training_data


class SmallMLP(nn.Module):
    """Small classifier for federated aggregation (same architecture on all sites)."""

    def __init__(self, n_features: int = 13, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def get_parameters(model: nn.Module) -> List[np.ndarray]:
    return [p.detach().numpy().copy() for p in model.parameters()]


def set_parameters(model: nn.Module, params: List[np.ndarray]) -> None:
    for p, new_p in zip(model.parameters(), params):
        p.data.copy_(torch.tensor(new_p))


def fed_avg(client_params: List[List[np.ndarray]], weights: List[float] = None) -> List[np.ndarray]:
    """Weighted average of client model parameters."""
    if weights is None:
        weights = [1.0 / len(client_params)] * len(client_params)
    avg = []
    for i in range(len(client_params[0])):
        avg.append(sum(w * cp[i] for w, cp in zip(weights, client_params)))
    return avg


def train_local(
    model: nn.Module,
    X: torch.Tensor,
    y: torch.Tensor,
    epochs: int = 10,
    lr: float = 0.01,
) -> None:
    optim = torch.optim.SGD(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        pred = model(X)
        loss = nn.functional.binary_cross_entropy(pred, y)
        optim.zero_grad()
        loss.backward()
        optim.step()


def evaluate(model: nn.Module, X: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        pred = (model(X) > 0.5).float()
        acc = (pred.squeeze() == y.squeeze()).float().mean().item()
    return acc


def run_federated(
    data_path: str,
    label_col: str = "label",
    num_sites: int = 3,
    rounds: int = 5,
    local_epochs: int = 10,
    n_features: int = 13,
) -> dict:
    """Simulate federated training: partition data by site, local training, FedAvg aggregation."""
    X, y = load_training_data(data_path, label_col=label_col)
    X = X.values.astype(np.float32)
    y = y.values.astype(np.float32).reshape(-1, 1)
    # Partition by row (simulate sites)
    indices = np.random.permutation(len(X))
    splits = np.array_split(indices, num_sites)
    site_data: List[Tuple[np.ndarray, np.ndarray]] = [
        (X[s], y[s]) for s in splits
    ]
    global_model = SmallMLP(n_features=n_features)
    for r in range(rounds):
        client_params = []
        for (Xi, yi) in site_data:
            if len(Xi) == 0:
                continue
            local_model = SmallMLP(n_features=n_features)
            set_parameters(local_model, get_parameters(global_model))
            X_t = torch.tensor(Xi, dtype=torch.float32)
            y_t = torch.tensor(yi, dtype=torch.float32)
            train_local(local_model, X_t, y_t, epochs=local_epochs)
            client_params.append(get_parameters(local_model))
        if client_params:
            new_global = fed_avg(client_params)
            set_parameters(global_model, new_global)
    # Final eval on full dataset
    X_all = torch.tensor(X, dtype=torch.float32)
    y_all = torch.tensor(y, dtype=torch.float32)
    acc = evaluate(global_model, X_all, y_all)
    return {"rounds": rounds, "num_sites": num_sites, "final_accuracy": acc}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Federated learning prototype (experimental)")
    parser.add_argument("--data", required=True, help="Path to CSV with feature columns + label")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--num-sites", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=10)
    args = parser.parse_args()
    result = run_federated(
        args.data,
        label_col=args.label_col,
        num_sites=args.num_sites,
        rounds=args.rounds,
        local_epochs=args.local_epochs,
    )
    print("Federated training (experimental) result:", result)
