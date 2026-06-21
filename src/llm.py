import yaml
from langchain_ollama import ChatOllama

with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)