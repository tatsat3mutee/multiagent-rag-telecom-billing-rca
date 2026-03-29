"""Quick test: verify Llama 3.3 70B works through LangChain + Groq."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.resolve()))

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import GROQ_API_KEY, LLM_MODEL

print(f"Testing model: {LLM_MODEL} via Groq")

llm = ChatGroq(
    model=LLM_MODEL,
    api_key=GROQ_API_KEY,
    temperature=0.1,
    timeout=30,
)

messages = [
    SystemMessage(content="You are a telecom billing expert."),
    HumanMessage(content="In one sentence, what causes zero-billing anomalies in telecom systems?"),
]

print("Calling LLM...")
response = llm.invoke(messages)
print(f"SUCCESS: {response.content}")
