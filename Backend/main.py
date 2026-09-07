from fastapi import FastAPI

from Backend.websocket.handler import websocket_endpoint

app = FastAPI()


@app.get("/health")
async def health():
    return {"status": "ok"}


app.add_api_websocket_route("/ws/voice", websocket_endpoint)