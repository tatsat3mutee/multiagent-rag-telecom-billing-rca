"""Quick smoke test: verify the configured LLM (Groq or Kimi) works."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from openai import OpenAI
from config import LLM_API_KEY, LLM_BASE_URL, LLM_MODEL, LLM_PROVIDER

print(f"Testing provider={LLM_PROVIDER} model={LLM_MODEL}")

if not LLM_API_KEY:
    raise SystemExit("No LLM key set. Add GROQ_API_KEY or KIMI_API_KEY to .env and retry.")

kwargs = {"api_key": LLM_API_KEY}
if LLM_BASE_URL:
    kwargs["base_url"] = LLM_BASE_URL
client = OpenAI(**kwargs)
resp = client.chat.completions.create(
    model=LLM_MODEL,
    temperature=0.1,
    timeout=30,
    messages=[
        {"role": "system", "content": "You are a telecom billing expert."},
        {"role": "user", "content": "In one sentence, what causes zero-billing anomalies in telecom systems?"},
    ],
)
print(f"SUCCESS: {resp.choices[0].message.content}")
