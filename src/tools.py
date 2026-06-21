import os
import subprocess
import urllib.request
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_tavily import TavilySearch
from dotenv import load_dotenv

load_dotenv()

# 1. DuckDuckGo Search Tool
ddg_tool = DuckDuckGoSearchRun(
    name="duckduckgo_search",
    description="DuckDuckGo",
)

# 2. Tavily Search Tool
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

tavily_tool = TavilySearch(
    name="tavily_search",
    description="Tavily-Search",
    max_results=3,
    search_depth="basic",
    include_answer=True,
    include_raw_content=False,
    tavily_api_key=TAVILY_API_KEY
)

@tool
def duckduckgo_search(query: str) -> str:
    """
    Search DuckDuckGo for general web results.
    """
    return ddg_tool.run(query)

@tool
def tavily_search(query: str) -> str:
    """
    Search Tavily for high-quality, LLM-optimized web results.
    """
    if not TAVILY_API_KEY:
        print("Warning: Tavily API Key not set. Defaulting to DuckDuckGo-Search.")
        return ddg_tool.run(query)
    return tavily_tool.run(query)

@tool
def long_int_multiply(num1: str, num2: str) -> str:
    """
    Multiply two arbitrarily large integers passed as strings.
    """
    if num1 == "0" or num2 == "0":
        return "0"

    n1, n2 = len(num1), len(num2)
    result = [0] * (n1 + n2)

    for i in range(n1 - 1, -1, -1):
        for j in range(n2 - 1, -1, -1):
            mul = int(num1[i]) * int(num2[j])
            p1, p2 = i + j, i + j + 1
            total = mul + result[p2]

            result[p2] = total % 10
            result[p1] += total // 10

    start = 0
    while start < len(result) and result[start] == 0:
        start += 1

    return "".join(map(str, result[start:]))

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
    """Write content to a local file."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

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

tools = [
    duckduckgo_search,
    tavily_search,
    long_int_multiply, 
    read_file,
    write_file,
    list_dir,
    search_code,
    run_cmd, 
    git_status,
    git_diff,
    fetch_url
]