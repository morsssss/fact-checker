"""
Stage 2: the web front end.

Usage:
    uv run server/web_server.py

Serves the page in web/ and, on /ws, bridges one browser's live microphone-of-
the-video audio into the same real-time pipeline (realtime_pipeline.py) that
audio_fact_checker.py uses for local files. The browser extracts audio from
the playing <video> itself (see web/app.js) and streams raw PCM chunks up;
this just forwards them to Mistral and pushes fact-check results back down as
JSON, the moment each one is ready.

The vector db (server/embeddings.npy, server/facts.json) is built automatically
on first use if it isn't already there - see fact_checker.load_vector_db().
"""

import json
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from mistralai.client.models import AudioFormat

from common import get_client
from fact_checker import load_vector_db
from realtime_pipeline import transcribe_and_fact_check

WEB_DIR = Path(__file__).parent.parent / "web"

app = FastAPI()

# Created once at startup and shared by every browser connection, rather than
# re-authenticating and re-loading the vector db for each one.
client = get_client()
fact_embeddings, facts = load_vector_db()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket) -> None:
    """Handle one browser's audio stream for the lifetime of its connection."""
    await websocket.accept()

    # The browser's AudioContext picks its own sample rate (usually 48000Hz);
    # its first message tells us what it chose so we can tell Mistral to expect
    # the same thing.
    hello = json.loads(await websocket.receive_text())
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=hello["sample_rate"])

    async def receive_pcm_chunks():
        try:
            while True:
                yield await websocket.receive_bytes()
        except WebSocketDisconnect:
            return

    async def send_result(result: dict) -> None:
        await websocket.send_json(result)

    await transcribe_and_fact_check(client, fact_embeddings, facts, receive_pcm_chunks(), audio_format, send_result)


# Mounted last so it only catches what the routes above don't (index.html, demo.mp4, ...).
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    # Railway (and most hosts) assign the port via $PORT and expect a bind on
    # all interfaces; 127.0.0.1:8001 remains the default for local dev.
    port = int(os.environ.get("PORT", 8001))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
