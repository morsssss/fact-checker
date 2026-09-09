"""Small helpers shared by the one-time embedding step and the fact-checker."""

import os

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()  # reads MISTRAL_API_KEY from a .env file at the repo root, if present

EMBEDDING_MODEL = "mistral-embed"
CHAT_MODEL = "mistral-medium-3.5"


def get_client() -> Mistral:
    """Build a Mistral client from the MISTRAL_API_KEY environment variable."""
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        raise RuntimeError("Set MISTRAL_API_KEY in a .env file (or the environment) before running this script.")
    return Mistral(api_key=api_key)


def load_facts(path: str) -> list[tuple[str, str]]:
    """Read a source-of-truth file into a list of (id, fact_text) pairs.

    Each non-blank line is one fact; its 1-based line number is its id.
    """
    facts = []
    with open(path) as f:
        for line_number, line in enumerate(f, start=1):
            text = line.strip()
            if text:
                facts.append((str(line_number), text))
    return facts
