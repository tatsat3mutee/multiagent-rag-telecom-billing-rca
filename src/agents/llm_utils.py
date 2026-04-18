"""
Shared LLM utility — single call_llm() used by all agents.
OpenAI backend with basic retry on transient errors.
"""
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_TEMPERATURE, LLM_PROVIDER

_client = None
_banner_shown = False


def _get_client():
    global _client, _banner_shown
    if _client is not None:
        return _client
    try:
        from openai import OpenAI
        if not LLM_API_KEY:
            _client = None
            return _client
        kwargs = {"api_key": LLM_API_KEY}
        if LLM_BASE_URL:
            kwargs["base_url"] = LLM_BASE_URL
        _client = OpenAI(**kwargs)
        if not _banner_shown:
            print(f"[LLM] provider={LLM_PROVIDER} model={LLM_MODEL}")
            _banner_shown = True
    except Exception as e:
        print(f"[LLM] client init failed: {e}")
        _client = None
    return _client


def call_llm(system_prompt: str, user_prompt: str, temperature: float = None,
             model: str = None, max_retries: int = 3) -> str:
    """
    Call the OpenAI LLM. Retries on 429/5xx/timeout.

    Returns the content string, or None on failure.
    """
    if temperature is None:
        temperature = LLM_TEMPERATURE
    if model is None:
        model = LLM_MODEL

    client = _get_client()
    if client is None:
        print("[LLM] No client available — set GROQ_API_KEY or KIMI_API_KEY in .env")
        return None

    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=60,
            )
            return resp.choices[0].message.content
        except Exception as e:
            msg = str(e).lower()
            retryable = any(k in msg for k in ("rate", "timeout", "502", "503", "504", "429"))
            if retryable and attempt < max_retries - 1:
                wait = 2 ** attempt
                print(f"[LLM] Retryable error ({e}); sleeping {wait}s")
                time.sleep(wait)
                continue
            print(f"[LLM] Call failed ({model}): {e}")
            return None
    return None
