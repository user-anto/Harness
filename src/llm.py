import yaml
from openai import OpenAI
from langchain_ollama import ChatOllama
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

orch_model = config.get("default_orch_model")
MODEL_CTX = config.get("ctx-size", 128) * 1024

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

search_llm = ChatOllama(
    model=config.get("default_search_model"),
    temperature=0.1
    )

guard_llm = ChatOllama(
    model=config.get("default_guard_model"),
    temperature=0.0
)