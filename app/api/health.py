from fastapi import APIRouter

from app.services import inference

router = APIRouter()


@router.get("/health")
def health():
    status = inference.get_device_status()
    return {
        "status": "ok" if status["model_loaded"] else "degraded",
        **status,
    }
