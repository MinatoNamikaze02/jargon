import hashlib
import json
import os
from typing import Dict, List, Optional
import concurrent.futures
import logging

from tenacity import retry, stop_after_attempt, wait_exponential
from openai import OpenAI

from .data import chunk_text_with_offsets


ANNOTATION_SYSTEM_PROMPT = (
    "You are an expert legal NER tagger.\n"
    "You are given text and a list of entity categories.\n"
    "Extract spans ONLY for the requested categories.\n"
    "Spans must be non-overlapping. You do NOT need to cover the entire text.\n"
    "Return a strict JSON object with key 'entities' mapping category -> list of spans.\n"
    "Each span must be {\"text\": str, \"start\": int, \"end\": int}.\n"
)

def _hash_key(text: str, entities: List[str], model: str) -> str:
    m = hashlib.sha1()
    m.update(text.encode("utf-8"))
    m.update("||".join(sorted(entities)).encode("utf-8"))
    m.update(model.encode("utf-8"))
    return m.hexdigest()


def _default_cache_dir() -> str:
    return os.path.join(os.getcwd(), ".cache", "annotations")


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


@retry(wait=wait_exponential(multiplier=1, min=1, max=10), stop=stop_after_attempt(3))
def _call_openai(client: OpenAI, model: str, text: str, entities: List[str]) -> Dict:
    categories = ", ".join(entities)
    user_prompt = (
        "Extract entity mentions from the text.\n"
        "Return a strict JSON object with key 'entities' mapping category -> list of spans.\n"
        "Each span must be {\"text\": str, \"start\": int, \"end\": int}.\n"
        f"Categories: {categories}.\n\nText:\n{text}"
    )
    resp = client.chat.completions.create(
        model=model,
        temperature=0,
        timeout=20,
        messages=[
            {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        response_format={"type": "json_object"},
    )
    content = resp.choices[0].message.content
    return json.loads(content)


def annotate_text(
    text: str,
    entities: List[str],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    cache_dir: Optional[str] = None,
) -> Dict[str, List[Dict]]:
    if client is None:
        client = OpenAI()
    if cache_dir is None:
        cache_dir = _default_cache_dir()
    _ensure_dir(cache_dir)

    cache_key = _hash_key(text, entities, model)
    cache_path = os.path.join(cache_dir, f"{cache_key}.json")
    if os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    result = _call_openai(client, model, text, entities)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False)
    return result


def annotate_long_text(
    text: str,
    entities: List[str],
    client: Optional[OpenAI] = None,
    model: str = "gpt-4o-mini",
    cache_dir: Optional[str] = None,
    max_chunk_tokens: int = 400,
    max_chunks: Optional[int] = 2,
    max_workers: int = 4,
) -> Dict[str, List[Dict]]:
    logger = logging.getLogger(__name__)
    chunks = chunk_text_with_offsets(text, max_tokens=max_chunk_tokens)
    if max_chunks is not None:
        chunks = chunks[:max_chunks]
    merged: Dict[str, List[Dict]] = {e: [] for e in entities}

    # Process chunks concurrently for speed
    def _process(idx_and_chunk):
        idx, (chunk_text_, base_offset) = idx_and_chunk
        try:
            out = annotate_text(chunk_text_, entities, client=client, model=model, cache_dir=cache_dir)
            logger.debug("annotated chunk %d (cache=%s)", idx, "hit" if out else "miss")
            return idx, base_offset, out
        except Exception as exc:
            logger.warning("annotation failed for chunk %d: %s", idx, exc)
            return idx, base_offset, {"entities": {}}

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(_process, (i, c)) for i, c in enumerate(chunks)]
        for fut in concurrent.futures.as_completed(futures):
            idx, base_offset, out = fut.result()
            for e in entities:
                spans = out.get("entities", {}).get(e, [])
                if isinstance(spans, list):
                    # Shift offsets back to full-document coordinates
                    for s in spans:
                        try:
                            s["start"] = int(s["start"]) + base_offset
                            s["end"] = int(s["end"]) + base_offset
                        except Exception:
                            continue
                    merged[e].extend(spans)
    return {"entities": merged}

