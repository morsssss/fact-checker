import os
from mistralai.client import Mistral
MISTRAL_API_KEY = os.getenv("MISTRAL_API_KEY")

AUDIO_FILE_PATH = 'assets/ben-babbles.m4a'

mistral = Mistral(api_key=MISTRAL_API_KEY)

with open(AUDIO_FILE_PATH, "rb") as f:
    transcription_response = mistral.audio.transcriptions.complete(
        model='voxtral-mini-latest',
        file={
            "content": f,
        }
            "file_name": "audio.mp3",
    )

