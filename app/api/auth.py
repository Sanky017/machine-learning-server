import os

from fastapi import Header, HTTPException

ADMIN_API_KEY = os.getenv("ADMIN_API_KEY")


def verify_admin(x_admin_key: str = Header(...)):
    """
    FastAPI dependency — attach to any route that should be admin-only:

        @router.get("/logs", dependencies=[Depends(verify_admin)])

    Client must send header:  X-Admin-Key: <the key>
    """
    if not ADMIN_API_KEY:
        # Fails closed: if the server forgot to set the env var, nobody
        # gets admin access rather than everybody getting it.
        raise HTTPException(status_code=500, detail="Server misconfigured: ADMIN_API_KEY not set")
    if x_admin_key != ADMIN_API_KEY:
        raise HTTPException(status_code=403, detail="Admin access required")
