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


def load_facts(path: str) -> list[dict]:
    """Read a source-of-truth file into fact records.

    Each non-blank line is tab-separated: fact text, then an optional source
    name, then an optional source URL. Its 1-based line number is its id.
    """
    facts = []
    with open(path) as f:
        for line_number, raw_line in enumerate(f, start=1):
            line = raw_line.rstrip("\n")
            if not line.strip():
                continue
            fields = line.split("\t")
            text = fields[0].strip()
            source = fields[1].strip() if len(fields) > 1 else ""
            url = fields[2].strip() if len(fields) > 2 else ""
            facts.append({"id": str(line_number), "text": text, "source": source, "url": url})
    return facts
