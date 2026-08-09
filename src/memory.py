import os
import uuid
import yaml
import ollama
import json
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
import atexit

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

with open(os.path.join(BASE_DIR, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)

EMBED_MODEL = config.get("embedding_model", "mxbai-embed-large:latest")
COLLECTION_NAME = "conversation_history"
QDRANT_PATH = os.path.join(BASE_DIR, "qdrant_db")

qdrant = QdrantClient(path=QDRANT_PATH)

def cleanup_qdrant():
    try:
        qdrant.close()
    except Exception:
        pass

atexit.register(cleanup_qdrant)

def _ensure_collection(dimension: int):
    if not qdrant.collection_exists(COLLECTION_NAME):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=dimension, distance=Distance.COSINE),
        )

def embed_and_save(prompt: str, response: str, trace_id: str, prompt_id: str, response_id: str):
    """Embed a prompt-response pair and save it to the Qdrant database."""
    text = f"User: {prompt}\nAssistant: {response}"
    
    chunk_size = 1500
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    
    try:
        points = []
        for chunk in chunks:
            emb_response = ollama.embeddings(model=EMBED_MODEL, prompt=chunk)
            embedding = emb_response["embedding"]
            
            _ensure_collection(len(embedding))
            
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=embedding,
                    payload={"trace_id": trace_id, "prompt_id": prompt_id, "response_id": response_id}
                )
            )
            
        if points:
            qdrant.upsert(
                collection_name=COLLECTION_NAME,
                points=points
            )
    except Exception as e:
        print(f"\n\033[91m[Archival Error] Failed to save to Qdrant: {str(e)}\033[0m")

def _fetch_log_content(trace_id: str, prompt_id: str, response_id: str) -> str:
    log_path = os.path.join(BASE_DIR, "execution_traces", trace_id)
    if not os.path.exists(log_path):
        return ""
    prompt_text = ""
    response_text = ""
    try:
        with open(log_path, "r") as f:
            for line in f:
                if not line.strip().startswith("{"):
                    continue
                try:
                    data = json.loads(line)
                    if data.get("timestamp") == prompt_id:
                        prompt_text = data.get("content", "")
                    elif data.get("timestamp") == response_id:
                        response_text = data.get("content", "")
                except Exception:
                    pass
    except Exception:
        pass
        
    if prompt_text and response_text:
        return f"User: {prompt_text}\nAssistant: {response_text}"
    return ""

def retrieve_context(query: str, top_k: int = 3) -> str:
    """Retrieve relevant conversation history from Qdrant based on the query."""
    try:
        emb_response = ollama.embeddings(model=EMBED_MODEL, prompt=query)
        embedding = emb_response["embedding"]
        
        if not qdrant.collection_exists(COLLECTION_NAME):
            return ""
            
        search_result = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=embedding,
            limit=top_k
        )
        
        if not search_result or not search_result.points:
            return ""
            
        context_blocks = []
        seen = set()
        for hit in search_result.points:
            trace_id = hit.payload.get("trace_id")
            prompt_id = hit.payload.get("prompt_id")
            response_id = hit.payload.get("response_id")
            
            if not (trace_id and prompt_id and response_id):
                # Fallback to old format if payload still has 'text'
                if "text" in hit.payload:
                    text_val = hit.payload["text"]
                    if text_val not in seen:
                        seen.add(text_val)
                        context_blocks.append(text_val)
                continue
                
            hit_key = (trace_id, prompt_id, response_id)
            if hit_key not in seen:
                seen.add(hit_key)
                content = _fetch_log_content(trace_id, prompt_id, response_id)
                if content:
                    context_blocks.append(content)
            
        return "\n\n---\n\n".join(context_blocks)
    except Exception as e:
        print(f"\n\033[91m[Retrieval Error] Failed to search Qdrant: {str(e)}\033[0m")
        return ""