"""
Shared LLM utility — single call_llm() used by all agents.
Prevents DRY violation by centralizing LLM access.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import GROQ_API_KEY, LLM_MODEL, LLM_TEMPERATURE


def call_llm(system_prompt: str, user_prompt: str, temperature: float = None) -> str:
    """
    Call the Groq LLM.

    Returns:
        The LLM response content string, or None if the call fails.
    """
    if temperature is None:
        temperature = LLM_TEMPERATURE

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import SystemMessage, HumanMessage
        llm = ChatGroq(
            model=LLM_MODEL,
            api_key=GROQ_API_KEY,
            temperature=temperature,
            timeout=30,
        )
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        response = llm.invoke(messages)
        return response.content
    except Exception as e:
        print(f"[LLM] Call failed ({LLM_MODEL}): {e}")
        return None
