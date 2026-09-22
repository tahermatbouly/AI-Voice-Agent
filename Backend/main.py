from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from Backend.websocket.handler import websocket_endpoint

app = FastAPI(title="GB Voice Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok"}


app.add_api_websocket_route("/ws/voice", websocket_endpoint)
