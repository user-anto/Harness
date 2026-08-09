import os
import subprocess
import urllib.request
from langchain_core.tools import tool
from dotenv import load_dotenv
load_dotenv()  # Load environment variables before setting up clients

from llm import config
import ollama
from memory import retrieve_context

@tool
def search_memory(query: str) -> str:
    """Search the user's conversational memory history. 
    CRITICAL: You MUST use this tool FIRST before performing any web searches to check if the topic has been discussed previously."""
    with Spinner("Digging through memory..."):
        result = retrieve_context(query)
        if not result:
            return "No relevant history found."
        return result

@tool
def long_int_multiply(num1: str, num2: str) -> str:
    """Multiply two arbitrarily large integers passed as strings."""
    return str(int(num1) * int(num2))

@tool
def read_file(path: str) -> str:
    """Read the contents of a local file."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(path: str, content: str) -> str:
    """Write content to a local file.
    If creating a new file:
        - place it in the 'workspace/' directory by default. (Use 'mkdir -p' command with the run_cmd tool to create the directory if needed.)
        - unless the user explicitly requests it in a certain specified directory."""
    abort = _ask_permission(f"\n\033[93mPermission to execute write_file on {path}? (Y/n, or provide feedback): \033[0m")
    if abort:
        return abort
    try:
        dir_name = os.path.dirname(os.path.abspath(path))
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def _ask_permission(prompt: str) -> str | None:
    """Return None if allowed, else abort reason."""
    ans = input(prompt).strip()
    low = ans.lower()
    if low in ("n", "no"):
        return "Execution aborted by user."
    if low not in ("", "y", "yes"):
        return f"Execution aborted. User provided feedback: {ans}"
    return None

@tool
def list_dir(path: str) -> str:
    """List the contents of a directory."""
    try:
        return "\n".join(os.listdir(path))
    except Exception as e:
        return f"Error listing directory: {str(e)}"

@tool
def search_code(directory: str, query: str) -> str:
    """Search for a string in a directory recursively using grep."""
    try:
        result = subprocess.run(["grep", "-rn", query, directory], capture_output=True, text=True)
        return result.stdout if result.stdout else "No matches found."
    except Exception as e:
        return f"Error searching code: {str(e)}"

@tool
def run_cmd(cmd: str, cwd: str = ".") -> str:
    """Run a shell command."""
    abort = _ask_permission(f"\n\033[93mPermission to execute command '{cmd}' in cwd '{cwd}'? (Y/n, or provide feedback): \033[0m")
    if abort:
        return abort
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        out = result.stdout + "\n" + result.stderr
        return out if out.strip() else "Command executed successfully with no output."
    except Exception as e:
        return f"Error running command: {str(e)}"

@tool
def git_status(repo_path: str = ".") -> str:
    """Get the git status of a repository."""
    try:
        result = subprocess.run("git status", shell=True, cwd=repo_path, capture_output=True, text=True)
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def git_diff(repo_path: str = ".") -> str:
    """Get the git diff of a repository."""
    try:
        result = subprocess.run("git diff", shell=True, cwd=repo_path, capture_output=True, text=True)
        return result.stdout + "\n" + result.stderr
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def fetch_url(url: str) -> str:
    """Fetch content from a URL via HTTP request."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            return response.read().decode('utf-8')
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

from cli import Spinner

@tool
def perform_web_search(query: str) -> str:
    """Perform a web search to gather information. 
    WARNING: You must call `search_memory` first before using this tool to ensure the information isn't already in the conversation history."""
    api_key = os.getenv("OLLAMA_API_KEY")
    if not api_key:
        return "Error: OLLAMA_API_KEY is not set in the .env file. An API key from ollama.com is required for web search."

    client = ollama.Client(
        headers={'Authorization': f'Bearer {api_key}'}
    )
    
    available_tools = {
        'web_search': client.web_search,
        'web_fetch': client.web_fetch
    }
    
    messages = [
        {'role': 'system', 'content': "You are a precise web search assistant. Your job is to search the web and return the exact information requested without any conversational filler or meta-commentary."},
        {'role': 'user', 'content': query}
    ]
    
    search_model = config.get("default_search_model", "llama3.2:3b")
    
    final_result = None
    with Spinner("Searching the web..."):
        while True:
            try:
                response = client.chat(
                    model=search_model,
                    messages=messages,
                    tools=[client.web_search, client.web_fetch],
                )
            except Exception as e:
                final_result = f"Error communicating with Ollama: {str(e)}"
                break
                
            messages.append(response.message)
            
            if response.message.tool_calls:
                for tool_call in response.message.tool_calls:
                    function_to_call = available_tools.get(tool_call.function.name)
                    if function_to_call:
                        try:
                            args = tool_call.function.arguments
                            result = function_to_call(**args)
                            truncated_result = str(result)[:8000]
                            messages.append({
                                'role': 'tool',
                                'content': truncated_result,
                                'tool_name': tool_call.function.name
                            })
                        except Exception as e:
                            messages.append({
                                'role': 'tool',
                                'content': f"Error calling tool: {str(e)}",
                                'tool_name': tool_call.function.name
                            })
                    else:
                        messages.append({
                            'role': 'tool',
                            'content': f"Tool {tool_call.function.name} not found",
                            'tool_name': tool_call.function.name
                        })
            else:
                final_result = response.message.content
                break

    print("\033[90mSearched the web\033[0m")
    return final_result

tools = [
    long_int_multiply, 
    read_file,
    write_file,
    list_dir,
    search_code,
    run_cmd, 
    git_status,
    git_diff,
    fetch_url,
    perform_web_search,
    search_memory
]