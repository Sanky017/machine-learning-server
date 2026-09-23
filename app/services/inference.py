"""
Inference service.

Deliberately has zero FastAPI/HTTP imports — you should be able to test
this from a plain Python shell:

    >>> from app.services.inference import predict
    >>> predict("CCO")
"""

import os
from pathlib import Path

import torch

from app.models.qsar_model import QSARNet
from app.services.featurize import InvalidMoleculeError, featurize_smiles

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TRAINED_MODELS_DIR = BASE_DIR / "trained_models"

_model = None
_feat_mean = None
_feat_std = None
_device = None


def _resolve_device() -> torch.device:
    override = os.getenv("DEVICE_OVERRIDE", "").strip().lower()
    if override in ("cuda", "cpu"):
        return torch.device(override)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_filename: str | None = None) -> None:
    """
    Load the trained model into memory. Call once at server startup.
    Safe to call again to hot-reload a new model version (used by
    /admin/reload-model).
    """
    global _model, _feat_mean, _feat_std, _device

    _device = _resolve_device()
    model_filename = model_filename or os.getenv("MODEL_FILENAME", "model_v1.pt")
    model_path = TRAINED_MODELS_DIR / model_filename

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found: {model_path}. "
            "Run app/models/train.py first, or check MODEL_FILENAME in .env."
        )

    checkpoint = torch.load(model_path, map_location=_device)

    model = QSARNet(input_dim=checkpoint["input_dim"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(_device)
    model.eval()

    _model = model
    _feat_mean = checkpoint["feat_mean"].to(_device)
    _feat_std = checkpoint["feat_std"].to(_device)

    print(f"Model loaded from {model_path} on device: {_device}")


def get_device_status() -> dict:
    return {
        "device": str(_device) if _device else "not_loaded",
        "cuda_available": torch.cuda.is_available(),
        "model_loaded": _model is not None,
    }


def predict(smiles: str) -> dict:
    """
    Run inference for a single SMILES string.
    Raises InvalidMoleculeError if the SMILES can't be parsed —
    the API layer should catch this and return HTTP 422, not 500.
    Raises RuntimeError if load_model() hasn't been called yet.
    """
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() at startup.")

    features = featurize_smiles(smiles)  # raises InvalidMoleculeError on bad input
    x = torch.tensor(features, dtype=torch.float32, device=_device).unsqueeze(0)
    x_norm = (x - _feat_mean) / _feat_std

    with torch.no_grad():
        prediction = _model(x_norm)

    return {
        "smiles": smiles,
        "prediction": float(prediction.item()),
        "device": str(_device),
    }
