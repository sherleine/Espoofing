import logging

from fastapi import FastAPI

from app.verification.router import router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)

app = FastAPI(title="Face verification and anti-spoofing")
app.include_router(router)
