import sqlite3
from pathlib import Path

from fastapi import APIRouter, Depends

from app.api.auth import verify_admin
from app.services import inference

router = APIRouter(prefix="/admin", dependencies=[Depends(verify_admin)])

DB_PATH = Path(__file__).resolve().parent.parent / "db" / "predictions.sqlite3"


@router.post("/reload-model")
def reload_model(model_filename: str | None = None):
    """Hot-reload the model without restarting the server (e.g. after retraining)."""
    inference.load_model(model_filename)
    return {"status": "reloaded", **inference.get_device_status()}


@router.get("/logs")
def view_logs(limit: int = 50):
    """Return the most recent logged predictions."""
    if not DB_PATH.exists():
        return {"logs": []}
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM predictions ORDER BY id DESC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return {"logs": [dict(row) for row in rows]}


@router.get("/status")
def status():
    """Full server status — device, model, useful for a quick admin sanity check."""
    return inference.get_device_status()
