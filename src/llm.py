import yaml
from openai import OpenAI
from langchain_ollama import ChatOllama

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

search_llm = ChatOllama(
    model=config.get("default_search_model"),
    temperature=0.1
    )

opt_llm = ChatOllama(
    model=config.get("default_opt_model"),
    temperature=0.2
    )

guard_llm = ChatOllama(
    model=config.get("default_guard_model"),
    temperature=0.0
)