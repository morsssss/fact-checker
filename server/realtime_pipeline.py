"""
Shared core of the real-time pipeline: stream audio into Mistral's realtime
transcription API, reassemble sentences, and fact-check each one concurrently.

Used by both audio_fact_checker.py (a local audio/video file) and
web_server.py (live audio streamed up from a browser).
"""

import asyncio
from typing import Awaitable, Callable, AsyncIterator

from mistralai.client.models import (
    AudioFormat,
    RealtimeTranscriptionError,
    TranscriptionStreamDone,
    TranscriptionStreamTextDelta,
)

from fact_checker import check_sentence, split_into_sentences

TRANSCRIPTION_MODEL = "voxtral-mini-transcribe-realtime-2602"

ResultCallback = Callable[[dict], Awaitable[None]]


def extract_complete_sentences(buffer: str, already_processed: int) -> tuple[list[str], int]:
    """Split the transcript-so-far into sentences, holding back a trailing partial one.

    We can't tell a finished sentence from a still-being-spoken one except by
    checking whether the buffer itself currently ends in sentence punctuation.
    Returns the newly-completed sentences and the updated processed count.
    """
    sentences = split_into_sentences(buffer)
    complete = sentences if buffer.rstrip()[-1:] in ".!?" else sentences[:-1]
    return complete[already_processed:], len(complete)


async def transcribe_and_fact_check(
    client,
    fact_embeddings,
    facts: list[dict],
    audio_stream: AsyncIterator[bytes],
    audio_format: AudioFormat,
    on_result: ResultCallback,
) -> None:
    """Stream audio_stream through Mistral's realtime transcription, fact-checking
    each sentence as it completes and calling on_result(result) for each one.

    Fact-checking a sentence takes several LLM calls - far longer than it takes
    for the next transcription event to arrive - so each one runs as its own
    concurrent task instead of blocking this loop (and the audio still streaming
    in behind it).
    """
    transcript_so_far = ""
    processed_count = 0
    in_flight: set[asyncio.Task] = set()  # holds refs so tasks aren't GC'd mid-flight

    async def fact_check_one(sentence: str) -> None:
        result = await check_sentence(client, fact_embeddings, facts, sentence)
        if result is not None:
            await on_result(result)

    async for event in client.audio.realtime.transcribe_stream(
        audio_stream, model=TRANSCRIPTION_MODEL, audio_format=audio_format
    ):
        if isinstance(event, TranscriptionStreamTextDelta):
            transcript_so_far += event.text
            new_sentences, processed_count = extract_complete_sentences(transcript_so_far, processed_count)
        elif isinstance(event, TranscriptionStreamDone):
            # Final flush: the API's own full transcript is authoritative, and every
            # sentence in it is now complete since there's no more audio coming.
            transcript_so_far = event.text
            new_sentences = split_into_sentences(transcript_so_far)[processed_count:]
        elif isinstance(event, RealtimeTranscriptionError):
            raise RuntimeError(f"Realtime transcription error: {event}")
        else:
            continue

        for sentence in new_sentences:
            task = asyncio.create_task(fact_check_one(sentence))
            in_flight.add(task)
            task.add_done_callback(in_flight.discard)

    if in_flight:
        await asyncio.gather(*in_flight)  # let any still-running fact-checks finish
