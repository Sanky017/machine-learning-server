from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI

load_dotenv()  # reads .env before anything else touches os.getenv

from app.api import admin, health, predict  # noqa: E402  (must come after load_dotenv)
from app.db.logging import init_db  # noqa: E402
from app.services import inference  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    init_db()
    try:
        inference.load_model()
    except FileNotFoundError as e:
        # Don't crash the whole server if no model is trained yet — /health
        # will report "degraded" and /predict will return 503 until a model
        # is loaded via /admin/reload-model.
        print(f"Warning: {e}")
    yield
    # --- shutdown --- (nothing to clean up currently)


app = FastAPI(title="MOF QSAR Inference Server", lifespan=lifespan)

app.include_router(predict.router)
app.include_router(health.router)
app.include_router(admin.router)
