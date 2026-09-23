"""
Training script — run this on the GPU workstation.

Usage:
    python -m app.models.train --data data/train.csv --version v1

Expects a CSV with at least two columns: `smiles` and `target`.
Replace `target` handling once your real QSAR label is finalized.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.models.qsar_model import QSARNet
from app.services.featurize import FEATURE_NAMES, featurize_batch

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"
METADATA_DIR = BASE_DIR / "metadata"
TRAINED_MODELS_DIR.mkdir(exist_ok=True)
METADATA_DIR.mkdir(exist_ok=True)


def load_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    import csv

    smiles_list, targets = [], []
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            smiles_list.append(row["smiles"])
            targets.append(float(row["target"]))

    features, valid_idx = featurize_batch(smiles_list)
    targets = np.array(targets, dtype=np.float32)[valid_idx]

    n_dropped = len(smiles_list) - len(valid_idx)
    if n_dropped:
        print(f"Warning: dropped {n_dropped} unparseable SMILES out of {len(smiles_list)}")

    return features, targets


def train(data_path: str, version: str, epochs: int = 100, lr: float = 1e-3, batch_size: int = 32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    X, y = load_dataset(data_path)
    if len(X) == 0:
        raise RuntimeError("No valid training samples after featurization — check your CSV.")

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

    # normalize features (save stats — inference must apply the same transform)
    feat_mean = X_tensor.mean(dim=0)
    feat_std = X_tensor.std(dim=0).clamp(min=1e-6)
    X_norm = (X_tensor - feat_mean) / feat_std

    dataset = TensorDataset(X_norm, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = QSARNet(input_dim=X.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    final_loss = None
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(dataset)
        final_loss = epoch_loss
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} — loss: {epoch_loss:.4f}")

    # --- save weights + normalization stats together ---
    model_filename = f"model_{version}.pt"
    save_path = TRAINED_MODELS_DIR / model_filename
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "input_dim": X.shape[1],
            "feat_mean": feat_mean,
            "feat_std": feat_std,
        },
        save_path,
    )

    # --- save metadata for traceability ---
    metadata = {
        "version": version,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_path": data_path,
        "n_samples": len(X),
        "feature_names": FEATURE_NAMES,
        "hyperparameters": {"epochs": epochs, "lr": lr, "batch_size": batch_size},
        "final_train_loss": final_loss,
        "device_used": str(device),
    }
    metadata_path = METADATA_DIR / f"model_{version}.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))

    print(f"Saved model to {save_path}")
    print(f"Saved metadata to {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to training CSV (columns: smiles, target)")
    parser.add_argument("--version", required=True, help="Version tag, e.g. v1")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    train(args.data, args.version, args.epochs, args.lr, args.batch_size)
