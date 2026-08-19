"""
FastAPI application entry point for the REST/dashboard side of the
project. This process is separate from the LiveKit agent worker
(agent/worker.py) -- this one only serves the dashboard's HTTP API
(list calls, call detail, audio playback). It does not handle any
live call audio itself.

Run with:
    uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app import db
from api.calls import router as calls_router

app = FastAPI(title="AI Voice Call Agent -- Dashboard API")

# Allows the React dev server (running on a different port, e.g. 5173)
# to call this backend during development. Tighten this to the actual
# frontend origin before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(calls_router, prefix="/api")


@app.on_event("startup")
def on_startup():
    db.init_db()


@app.get("/health")
def health_check():
    return {"status": "ok"}