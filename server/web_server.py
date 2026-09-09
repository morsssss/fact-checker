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

import asyncio
import json
import os
import shutil
import tempfile
from pathlib import Path

import imageio_ffmpeg
import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from mistralai.client.models import AudioFormat

from common import get_client
from fact_checker import load_vector_db
from realtime_pipeline import transcribe_and_fact_check

WEB_DIR = Path(__file__).parent.parent / "web"
VIDEO_PATH = WEB_DIR / "video" / "claims.mp4"

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


@app.post("/upload-video")
async def upload_video(file: UploadFile = File(...)) -> dict:
    """Replace the demo video with an uploaded one.

    Saves the upload, re-encodes it with ffmpeg (downscaled to a max width of
    1280px if it's larger, moderate quality loss) so oversized phone-camera
    uploads don't balloon the page's load time, then swaps it in for
    web/video/claims.mp4 - the exact file <video> already points to.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        upload_path = Path(tmp_dir) / "upload"
        with upload_path.open("wb") as f:
            shutil.copyfileobj(file.file, f)

        compressed_path = Path(tmp_dir) / "compressed.mp4"
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        process = await asyncio.create_subprocess_exec(
            ffmpeg_exe,
            "-i", str(upload_path),
            "-vf", "scale='min(1280,iw)':-2",
            "-c:v", "libx264",
            "-preset", "medium",
            "-crf", "23",
            "-pix_fmt", "yuv420p",
            "-c:a", "aac",
            "-b:a", "128k",
            "-movflags", "+faststart",
            "-y",
            "-loglevel", "error",
            str(compressed_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate()

        if process.returncode != 0 or not compressed_path.exists():
            message = stderr.decode(errors="replace")[:500] or "ffmpeg could not process that file."
            raise HTTPException(status_code=400, detail=message)

        VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(compressed_path), VIDEO_PATH)

    return {"ok": True}


# Mounted last so it only catches what the routes above don't (index.html, demo.mp4, ...).
app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")


if __name__ == "__main__":
    # Railway (and most hosts) assign the port via $PORT and expect a bind on
    # all interfaces; 127.0.0.1:8001 remains the default for local dev.
    port = int(os.environ.get("PORT", 8001))
    host = "0.0.0.0" if "PORT" in os.environ else "127.0.0.1"
    uvicorn.run(app, host=host, port=port)
