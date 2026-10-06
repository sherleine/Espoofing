# server.py - run the prototype on its own:  uvicorn server:app --port 8000
# In the real application, just include app.verification.router.router.
import logging

from fastapi import FastAPI

from app.verification.router import router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

app = FastAPI(title="Face verification & anti-spoofing (prototype)")
app.include_router(router)
