"""
Stage 0: check a static transcript's factual assertions against the source of truth.

Usage:
    uv run server/fact_checker.py [path/to/transcript.txt]

This is an offline stand-in for the real-time pipeline described in spec.md:
instead of streaming audio in, we already have a full transcript file. For
each sentence we ask the LLM whether it's a factual assertion; if so, we embed
it, find the closest facts in the source-of-truth vector db (built by
process_source_of_truth.py), and ask the LLM to judge whether the assertion is
true.

The vector db (server/embeddings.npy, server/facts.json) is built automatically
on first use if it isn't already there - see load_vector_db() below.
"""

import asyncio
import json
import os
import re
import sys

import numpy as np

from common import CHAT_MODEL, EMBEDDING_MODEL, get_client

DEFAULT_TRANSCRIPT_PATH = "assets/test-transcription-8-25-26.txt"
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings.npy")
FACTS_INDEX_PATH = os.path.join(os.path.dirname(__file__), "facts.json")

# Below this cosine similarity, we treat the closest fact as unrelated rather
# than as a match - the source of truth simply has nothing to say about it.
MATCH_THRESHOLD = 0.5

TOP_K_MATCHES = 1

# Split transcript text into sentences wherever end-of-sentence punctuation is
# followed by whitespace. Stage 2 (streaming) will apply this same regex to
# each new chunk of live transcript as it arrives.
SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")


def split_into_sentences(text: str) -> list[str]:
    """Collapse a transcript into a flat list of sentences."""
    flat_text = " ".join(text.split())  # blank lines/wrapping -> single spaces
    sentences = SENTENCE_END_RE.split(flat_text)
    return [s.strip() for s in sentences if s.strip()]


async def detect_assertion(client, sentence: str) -> str | None:
    """Ask the LLM whether a sentence contains a factual assertion.

    Returns the assertion text if so, otherwise None.
    """
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You detect factual assertions in one sentence of a spoken transcript. "
                    'Reply with JSON: {"contains_assertion": bool, "assertion_text": string}. '
                    "assertion_text is the specific factual claim (quoted or paraphrased from "
                    "the sentence), or an empty string if there isn't one. Greetings, opinions, "
                    "and questions are not factual assertions."
                ),
            },
            {"role": "user", "content": sentence},
        ],
    )
    result = json.loads(response.choices[0].message.content)
    if result.get("contains_assertion"):
        return result.get("assertion_text") or sentence
    return None


def find_closest_facts(
    assertion_embedding: np.ndarray, fact_embeddings: np.ndarray, facts: list[dict], top_k: int = TOP_K_MATCHES
) -> list[tuple[dict, float]]:
    """Return the top_k facts closest to an assertion, ranked by cosine similarity."""
    a = assertion_embedding / np.linalg.norm(assertion_embedding)
    f = fact_embeddings / np.linalg.norm(fact_embeddings, axis=1, keepdims=True)
    similarities = f @ a
    top_indices = np.argsort(similarities)[::-1][:top_k]
    return [(facts[i], float(similarities[i])) for i in top_indices]


async def judge_assertion(client, assertion: str, candidate_facts: list[dict]) -> dict:
    """Ask the LLM to weigh an assertion against candidate facts and return a verdict."""
    facts_block = "\n".join(f"- {fact['text']}" for fact in candidate_facts)
    response = await client.chat.complete_async(
        model=CHAT_MODEL,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "You fact-check a claim against a list of known facts. Reply with JSON: "
                    '{"verdict": "true"|"false"|"likely_true"|"likely_false", "reasoning": string}. '
                    'Use "likely_*" only when the facts do not fully settle the claim.'
                ),
            },
            {"role": "user", "content": f"Claim: {assertion}\n\nKnown facts:\n{facts_block}"},
        ],
    )
    return json.loads(response.choices[0].message.content)


def load_vector_db() -> tuple[np.ndarray, list[dict]]:
    """Load the fact embeddings and their id/text metadata, building them first if needed.

    embeddings.npy/facts.json are gitignored (they're regenerable from assets/facts.txt),
    so a fresh checkout or deploy won't have them yet - building on first use here means
    no separate manual step is needed before running any entry point.
    """
    if not os.path.exists(EMBEDDINGS_PATH):
        import process_source_of_truth

        process_source_of_truth.build_vector_db()
    fact_embeddings = np.load(EMBEDDINGS_PATH)
    with open(FACTS_INDEX_PATH) as f:
        facts = json.load(f)
    return fact_embeddings, facts


async def check_sentence(client, fact_embeddings: np.ndarray, facts: list[dict], sentence: str) -> dict | None:
    """Run one sentence through the assertion/match/verdict pipeline.

    Returns a result dict, or None if the sentence contains no factual assertion.
    Shared by the offline (check_transcript) and real-time (audio_fact_checker.py) entry points.
    Uses the SDK's async methods throughout so a caller can run many of these
    concurrently (via asyncio.create_task) instead of one at a time.
    """
    assertion = await detect_assertion(client, sentence)
    if assertion is None:
        return None

    embedding_response = await client.embeddings.create_async(model=EMBEDDING_MODEL, inputs=[assertion])
    assertion_embedding = np.array(embedding_response.data[0].embedding)
    matches = find_closest_facts(assertion_embedding, fact_embeddings, facts)

    best_fact, best_score = matches[0]
    if best_score < MATCH_THRESHOLD:
        return {"text": assertion, "matched": False}

    verdict = await judge_assertion(client, assertion, [fact for fact, _ in matches])
    return {
        "text": assertion,
        "matched": True,
        "verdict": verdict["verdict"],
        "reasoning": verdict["reasoning"],
        "matches": [{"fact": fact["text"], "probability": round(score, 3)} for fact, score in matches],
    }


async def check_transcript(client, transcript_path: str) -> list[dict]:
    """Run the full Stage 0 pipeline and return one result per factual assertion found."""
    fact_embeddings, facts = load_vector_db()

    with open(transcript_path) as f:
        sentences = split_into_sentences(f.read())

    results = []
    for sentence in sentences:
        result = await check_sentence(client, fact_embeddings, facts, sentence)
        if result is not None:
            results.append(result)
    return results


async def main():
    transcript_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TRANSCRIPT_PATH
    client = get_client()
    results = await check_transcript(client, transcript_path)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
