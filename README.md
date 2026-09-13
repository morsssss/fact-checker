# Fact Checker

A real-time fact-checking demo!

We play a video. As the video plays, we extracts the audio and transcribe it in
real time. 

As the transcription streams in, we check it for sentences that look like
factual claims. We check each such claim against a source of truth and display
the result.

## How it works

1. The browser extracts audio from the playing `<video>` and streams it to
   our server over a WebSocket.
2. The server forwards audio to Mistral's real-time transcription model and
   watches the transcript for complete sentences.
3. Each sentence is checked for a factual assertion. If it contains one, the
   assertion is embedded and matched against embeddings we've already calculated
   for each fact in `assets/facts.tsv`.
4. If a close-enough match is found, an LLM judges the assertion against it
   and returns a verdict + reasoning, which is pushed back to the browser and
   rendered as a card.

## Requirements

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) for package management
- A Mistral API key

## Setup

1. Clone the repo.
2. Create a `.env` file in the repo root that contains your Mistral API key.
   ```
   MISTRAL_API_KEY=your-key-here
   ```
3. Install dependencies:
   ```
   uv sync
   ```
   (this also happens automatically the first time you `uv run` anything, so
   this step is optional)

There's no separate build step for the fact database. `server/embeddings.npy`
and `server/facts.json` are generated automatically from `assets/facts.tsv`
the first time any entry point runs. To rebuild them by hand (e.g. after
editing `assets/facts.tsv`), delete those two files or run:
```
uv run server/process_source_of_truth.py
```

## Running

### The web app

```
uv run server/web_server.py
```

Then open http://127.0.0.1:8001. Fact-checking starts when the video does.

### CLI tools

These are useful for testing pieces of the pipeline in isolation, without a
browser:

- `uv run server/fact_checker.py [path/to/transcript.txt]` - run a saved
  transcript through the fact-checking pipeline offline, no audio required.
- `uv run server/audio_fact_checker.py [path/to/audio-or-video-file]` -
  decode a local audio/video file and fact-check it in real time (paced at
  real playback speed), printing each result to the terminal as it's ready.

## The source of truth

`assets/facts.tsv` is tab-separated, with one fact per line. Each line contains:
* fact text
* an optional source name
* an optional source URL

When a fact with a source is shown in the UI, the source is displayed after
fact. If we have a URL, that source is a link. Edit this file and rebuild the vector db
(see Setup above) to change what the app checks claims against.

## Using other models
In the repo, this demo uses:
* `mistral-embed` to calculate embeddings
* `mistral-medium-3.5` for reasoning
* `voxtral-mini-transcribe-realtime-2602` for transcription

To change either of the first two models, change the model names at the top of `fact_checker.py`. To change the transcription model, change `TRANSCRIPTION_MODEL` in `realtime_pipeline.py`.

## Using other APIs
* Mistral conveniently provides all the models we need, but you could switch these out for another provider. It's also instructive to see how different models handle the fact-checking.


## Project layout

```
server/
  common.py                    shared helpers: Mistral client, source-of-truth loader
  fact_checker.py              the core pipeline; also runnable standalone (see its docstring)
  realtime_pipeline.py         streaming transcription + fact-checking, shared by both real-time entry points
  audio_fact_checker.py        CLI to fact-check a local audio/video file
  process_source_of_truth.py   builds the fact embeddings vector db
  web_server.py                FastAPI app: serves web/, bridges browser audio over a WebSocket

web/                           front end - vanilla HTML/CSS/JS, no build step
  index.html                   page markup: video, results panel, header controls
  style.css                    all styling - self-hosted Mistral brand fonts + colors, dark mode
  app.js                       client-side logic: captures audio from the <video>, streams it to
                                 the server over a WebSocket, renders results as they arrive
  audio-processor.js           an AudioWorkletProcessor - runs on the audio thread, turns captured
                                 audio into PCM chunks for app.js to send
  fonts/                       self-hosted Alt-Mistral + Inter font files, used by style.css
  img/                         logo and footer artwork
  video/claims.mp4             the default demo video played by index.html

assets/facts.tsv              the source of truth
```

## Deploying

A `Dockerfile` is included. It expects `MISTRAL_API_KEY` as an environment
variable at runtime:

```
docker build -t fact-checker .
docker run -p 8001:8001 -e MISTRAL_API_KEY=your-key-here fact-checker
```

`railway.json` configures the same app for deployment on Railway.
