"""O-Ton Stack — LLM-Schicht: DeepSeek V4 Flash (API) + Mistral Embed"""
import os
import json
import urllib.request

DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_URL = "https://api.deepseek.com/chat/completions"
LLM_MODEL = os.environ.get("LLM_MODEL", "deepseek-v4-flash")

MISTRAL_KEY = os.environ.get("MISTRAL_API_KEY", "")
MISTRAL_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"


def _post(url, payload, key):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:400]
        raise RuntimeError(f"LLM HTTP {e.code}: {body}") from e


def llm_chat(system: str, user: str, max_tokens: int = 700) -> str:
    d = _post(
        DEEPSEEK_URL,
        {
            "model": LLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        DEEPSEEK_KEY,
    )
    return d["choices"][0]["message"]["content"].strip()


def embed_text(text: str) -> list:
    d = _post(
        MISTRAL_URL,
        {"model": EMBED_MODEL, "input": text},
        MISTRAL_KEY,
    )
    return d["data"][0]["embedding"]
