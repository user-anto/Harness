import yaml
from openai import OpenAI

# Load configuration
with open("../config.yaml", "r") as f:
    config = yaml.safe_load(f)
orch_model = config.get("default_orch_model")

# Initialize OpenAI client pointing to the custom local endpoint
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="dummy"
)

def test_llm_call(prompt: str):
    """
    Directly invokes the LLM using the OpenAI SDK, streaming the response,
    and returns a tuple containing (reasoning, final_answer).
    """
    print(f"Prompt: {prompt}\n")
    
    response = client.chat.completions.create(
        model=orch_model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        stream=True
    )
    
    reasoning_parts = []
    content_parts = []
    
    print("--- Streaming Response ---")
    for chunk in response:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        
        # Extract reasoning content (if supported by the model/endpoint API)
        reasoning = getattr(delta, "reasoning_content", None)
        if reasoning:
            reasoning_parts.append(reasoning)
            print(reasoning, end="", flush=True)
            
        # Extract standard text content
        content = getattr(delta, "content", None)
        if content:
            content_parts.append(content)
            print(content, end="", flush=True)
            
    print("\n--------------------------\n")
    
    full_reasoning = "".join(reasoning_parts)
    full_content = "".join(content_parts)
    
    return full_reasoning, full_content

if __name__ == "__main__":
    prompt = "Why does C++ require a custom implementation for very large integer multiplication?"
    reasoning, answer = test_llm_call(prompt)
    
    print(f"Extracted Reasoning:\n{reasoning}\n")
    print(f"Extracted Final Answer:\n{answer}\n")
