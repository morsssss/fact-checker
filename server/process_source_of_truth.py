"""
One-time step: turn a source-of-truth file into a vector database.

Usage:
    uv run server/process_source_of_truth.py [path/to/facts.txt]

Reads a source-of-truth file (one fact per line, formatted "<id>\t<fact text>"),
embeds each fact with Mistral's embedding model, and writes:
  * server/embeddings.npy - the embedding vectors, one row per fact
  * server/facts.json     - the id/text for each row, in the same order

fact_checker.py reads both files back to match assertions against facts.
"""

import json
import os
import sys

import numpy as np

from common import EMBEDDING_MODEL, get_client, load_facts

DEFAULT_FACTS_PATH = "assets/facts.txt"
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npy")
FACTS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "facts.json")


def embed_facts(client, facts: list[tuple[str, str]]) -> np.ndarray:
    """Embed every fact's text in a single batched API call."""
    texts = [text for _, text in facts]
    response = client.embeddings.create(model=EMBEDDING_MODEL, inputs=texts)
    return np.array([row.embedding for row in response.data])


def build_vector_db(facts_path: str = DEFAULT_FACTS_PATH) -> None:
    """Embed facts_path and write embeddings.npy/facts.json. The actual work,
    independent of argv - fact_checker.load_vector_db() calls this directly
    (not main()) so it isn't affected by whatever argv the calling script was
    invoked with.
    """
    facts = load_facts(facts_path)
    if not facts:
        raise RuntimeError(f"No facts found in {facts_path}")

    client = get_client()
    embeddings = embed_facts(client, facts)

    np.save(EMBEDDINGS_PATH, embeddings)
    with open(FACTS_INDEX_PATH, "w") as f:
        json.dump([{"id": fact_id, "text": text} for fact_id, text in facts], f, indent=2)

    print(f"Embedded {len(facts)} facts -> {EMBEDDINGS_PATH} (+ {FACTS_INDEX_PATH})")


def main():
    facts_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_FACTS_PATH
    build_vector_db(facts_path)


if __name__ == "__main__":
    main()
