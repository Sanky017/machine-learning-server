# MOF QSAR Inference Server

GPU-backed FastAPI server for serving a QSAR model over REST. Training runs
on CUDA when available and falls back to CPU automatically.

## Architecture

```
Client → FastAPI (/predict) → featurize.py (RDKit) → QSARNet (PyTorch) → GPU/CPU → prediction
                                                                              ↓
                                                                    SQLite (logging)
```

- `app/services/featurize.py` — single source of truth for feature computation (used by both training and inference, avoids train/serve skew)
- `app/models/train.py` — training script, run on the GPU workstation
- `app/services/inference.py` — loads model once, exposes `predict()`, device-aware
- `app/api/` — FastAPI routes: `/predict` and `/health` are public, `/admin/*` requires an API key

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env              # then edit ADMIN_API_KEY in .env
```

On the GPU workstation, verify CUDA first:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

## Train

```bash
python -m app.models.train --data data/train.csv --version v1
```
CSV must have `smiles` and `target` columns. Produces:
- `trained_models/model_v1.pt`
- `metadata/model_v1.json`

Set `MODEL_FILENAME=model_v1.pt` in `.env` to match.

## Run

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 4
```
(`--workers` ~ 2x CPU cores. Drop to `--workers 1 --reload` while developing.)

## Test

```bash
# Public — no key needed
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"smiles": "CCO"}'

curl http://localhost:8000/health

# Admin — requires key from .env
curl http://localhost:8000/admin/status -H "X-Admin-Key: <your-key>"
curl http://localhost:8000/admin/logs -H "X-Admin-Key: <your-key>"
curl -X POST http://localhost:8000/admin/reload-model -H "X-Admin-Key: <your-key>"
```

## Notes

- CPU fallback is automatic (`torch.device("cuda" if available else "cpu")`) — the server won't crash if run on a machine without a GPU, it'll just be slower.
- Feature vectors are cached in `app/db/feature_cache/` keyed by SMILES hash to avoid recomputing RDKit descriptors for repeat inputs.
- For MOF-specific structural descriptors (pore size, surface area via pymatgen/Zeo++), extend `featurize.py` with a `featurize_structure()` function and combine with `featurize_smiles()` — keep both training and inference importing the same function.
