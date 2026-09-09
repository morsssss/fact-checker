"""
Stage 1: transcribe a local audio/video file in real time and fact-check it as
it's spoken.

Usage:
    uv run server/audio_fact_checker.py [path/to/audio-file]

Decodes the input file to raw PCM with ffmpeg and streams it into the shared
real-time pipeline (realtime_pipeline.py) paced at real playback speed, as if
it were a live feed. Each fact-check result is printed the moment it's ready.

The vector db (server/embeddings.npy, server/facts.json) is built automatically
on first use if it isn't already there - see fact_checker.load_vector_db().
"""

import asyncio
import json
import subprocess
import sys
from datetime import datetime
from typing import AsyncIterator

import imageio_ffmpeg
from mistralai.client.models import AudioFormat

from common import get_client
from fact_checker import load_vector_db
from realtime_pipeline import transcribe_and_fact_check

DEFAULT_AUDIO_PATH = "assets/ben-babbles.m4a"

SAMPLE_RATE = 16000
CHUNK_MS = 200  # how much audio we send per websocket message
CHUNK_BYTES = SAMPLE_RATE * 2 * CHUNK_MS // 1000  # 2 bytes/sample (16-bit PCM)


async def stream_pcm_chunks(audio_path: str) -> AsyncIterator[bytes]:
    """Decode any audio file to mono 16kHz PCM16 chunks, paced like real playback.

    ffmpeg (bundled via imageio_ffmpeg, no system install needed) handles decoding
    so this works with any input format - m4a, wav, audio extracted from video, etc.
    """
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    process = await asyncio.create_subprocess_exec(
        ffmpeg_exe,
        "-i", audio_path,
        "-f", "s16le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "-loglevel", "error",
        "pipe:1",
        stdout=subprocess.PIPE,
    )
    try:
        while True:
            chunk = await process.stdout.read(CHUNK_BYTES)
            if not chunk:
                break
            yield chunk
            await asyncio.sleep(CHUNK_MS / 1000)  # simulate real-time playback
    finally:
        await process.wait()


async def print_result(result: dict) -> None:
    """Print a fact-check result with a timestamp, the moment it's ready.

    flush=True: stdout is block-buffered when it isn't a terminal, so without
    this, prints can sit in the buffer until the process exits regardless of
    when the work actually finished - making everything look batched at the end.
    """
    print(datetime.now().isoformat(timespec="milliseconds"), flush=True)
    print(json.dumps(result, indent=2), flush=True)


async def run(audio_path: str) -> None:
    client = get_client()
    fact_embeddings, facts = load_vector_db()
    audio_format = AudioFormat(encoding="pcm_s16le", sample_rate=SAMPLE_RATE)
    await transcribe_and_fact_check(
        client, fact_embeddings, facts, stream_pcm_chunks(audio_path), audio_format, print_result
    )


def main():
    audio_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_AUDIO_PATH
    asyncio.run(run(audio_path))


if __name__ == "__main__":
    main()
