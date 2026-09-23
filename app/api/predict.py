from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services import inference
from app.services.featurize import InvalidMoleculeError
from app.db.logging import log_prediction

router = APIRouter()


class PredictRequest(BaseModel):
    smiles: str = Field(..., min_length=1, description="SMILES string of the molecule")


class PredictResponse(BaseModel):
    smiles: str
    prediction: float
    device: str


@router.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest):
    try:
        result = inference.predict(payload.smiles)
    except InvalidMoleculeError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except RuntimeError as e:
        # e.g. model not loaded — server-side setup problem, not the caller's fault
        raise HTTPException(status_code=503, detail=str(e))

    log_prediction(result["smiles"], result["prediction"], result["device"])
    return result
