"""
Featurization service.

IMPORTANT: This is the ONLY place feature vectors get computed — both
train.py and inference.py must import from here. If you compute features
differently in two places, you get train/serve skew and your model will
silently produce wrong predictions.

Current implementation: RDKit descriptors from a SMILES string.
TODO (MOF-specific): if your target needs structural descriptors (pore size,
surface area, etc.), add a `featurize_structure(cif_path)` function here
using pymatgen / Zeo++ and combine its output with `featurize_smiles`.
"""

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

# ---- simple on-disk cache (avoids recomputing descriptors for repeat inputs) ----
CACHE_DIR = Path(__file__).resolve().parent.parent / "db" / "feature_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# The fixed, ordered list of descriptors this pipeline produces.
# Keep this list stable — if you change it, retrain the model, don't just
# edit inference behaviour.
FEATURE_NAMES = [
    "MolWt",
    "MolLogP",
    "NumHAcceptors",
    "NumHDonors",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "FractionCSP3",
    "HeavyAtomCount",
]


class InvalidMoleculeError(ValueError):
    """Raised when a SMILES string cannot be parsed by RDKit."""


def _cache_key(smiles: str) -> Path:
    digest = hashlib.sha256(smiles.encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def featurize_smiles(smiles: str, use_cache: bool = True) -> np.ndarray:
    """
    Convert a SMILES string into a fixed-length numeric feature vector.

    Raises InvalidMoleculeError if the SMILES string is not parseable —
    callers (API layer) should catch this and return a 4xx, not a 500.
    """
    cache_path = _cache_key(smiles)
    if use_cache and cache_path.exists():
        cached = json.loads(cache_path.read_text())
        return np.array(cached["features"], dtype=np.float32)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise InvalidMoleculeError(f"Could not parse SMILES: {smiles!r}")

    features = [
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumRotatableBonds(mol),
        rdMolDescriptors.CalcNumRings(mol),
        rdMolDescriptors.CalcNumAromaticRings(mol),
        rdMolDescriptors.CalcFractionCSP3(mol),
        mol.GetNumHeavyAtoms(),
    ]
    feature_vec = np.array(features, dtype=np.float32)

    if use_cache:
        cache_path.write_text(json.dumps({"smiles": smiles, "features": feature_vec.tolist()}))

    return feature_vec


def featurize_batch(smiles_list: list[str]) -> tuple[np.ndarray, list[int]]:
    """
    Featurize a batch of SMILES strings.
    Returns (feature_matrix, valid_indices) — valid_indices tells you which
    input rows survived parsing, so you can align back to original data
    when some SMILES in a training set are malformed.
    """
    rows = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        try:
            rows.append(featurize_smiles(smi))
            valid_indices.append(i)
        except InvalidMoleculeError:
            continue
    if not rows:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float32), []
    return np.vstack(rows), valid_indices
