"""O-Ton Stack — LLM-Schicht: GLM-5.3-Flash (Z.AI) + Mistral Embed"""
import os
import json
import urllib.request

ZAI_KEY = os.environ.get("ZAI_API_KEY", "")
ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"
GLM_MODEL = os.environ.get("GLM_MODEL", "glm-5.3-flash")

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
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read().decode())


def llm_chat(system: str, user: str, max_tokens: int = 700) -> str:
    d = _post(
        ZAI_URL,
        {
            "model": GLM_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "temperature": 0.3,
            "max_tokens": max_tokens,
        },
        ZAI_KEY,
    )
    return d["choices"][0]["message"]["content"].strip()


def embed_text(text: str) -> list:
    d = _post(
        MISTRAL_URL,
        {"model": EMBED_MODEL, "inputs": [text]},
        MISTRAL_KEY,
    )
    return d["data"][0]["embedding"]
