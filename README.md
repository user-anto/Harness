<p align="center">
  <img src="logo.png" alt="Harness Logo" width="800">
</p>

<h1 align="center">Harness</h1>

Harness is a locally-hosted, agentic CLI framework built using LangGraph, LangChain, and Ollama. It provides an autonomous environment running directly inside the terminal, capable of managing context, searching the web, executing code, and editing files, with a design emphasizing security, auditability, and execution containment.

---

## Technical Features

```mermaid
graph TD
    %% User Input Flow
    User([User Input]) --> InputNode[Input Node]
    
    %% Guardrail Security
    InputNode --> Guardrail{Guardrail Node\n+ Base64/Hex Decoder}
    Guardrail -- Injection Detected --> Reject[Reject Request & Log Error]
    Guardrail -- Safe Input --> Router{Input Router}
    
    Router -- "/plan detected" --> PlannerNode[Planner Node]
    Router -- "Default Input" --> OrchNode[Orchestrator Node]
    PlannerNode -- "Injects Task List" --> OrchNode
    
    %% Main LLM Engine
    OrchNode -- "Streams Prompt & History" --> LLM[(Local llama-server)]
    
    %% Routing
    OrchNode -- "Decides to Call Tool" --> ToolNode[Tool Node]
    OrchNode -- "Streams Final Answer" --> ArchivalNode[Archival Node]
    
    %% Tool Execution Sub-Graph
    ToolNode --> Tools{Available Tools}
    
    Tools -- "search_memory" --> MemSearch[Memory Search]
    Tools -- "perform_web_search" --> WebSearch[Web Search Sub-Agent]
    Tools -- "run_cmd, write_file, etc." --> SysTools[OS / File System]
    
    %% Data Stores & External Dependencies
    MemSearch -- "Semantic Similarity" --> Qdrant[(Qdrant Vector DB)]
    MemSearch -- "Dynamic Payload Fetch" --> Traces[(execution_traces JSONL)]
    WebSearch -- "web_search & web_fetch" --> OllamaAPI[(Ollama API)]
    
    %% Tool Output Loop
    ToolNode -- "Tool Execution Results" --> OrchNode
    
    %% Archival & Cleanup
    ArchivalNode -- "1. Enforce Token Window\n2. Track Active\nConversation History" --> LangGraphState[(LangGraph State)]
    ArchivalNode -- "Async Embed & Save" --> EmbedSave[Background Thread]
    EmbedSave --> Qdrant
    
    %% Styling
    classDef primary fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff;
    classDef storage fill:#50E3C2,stroke:#333,stroke-width:2px,color:#000;
    classDef security fill:#E94B3C,stroke:#333,stroke-width:2px,color:#fff;
    classDef tool fill:#F5A623,stroke:#333,stroke-width:2px,color:#fff;
    
    class InputNode,Router,PlannerNode,OrchNode,ArchivalNode,ToolNode primary;
    class LLM,Qdrant,Traces,LangGraphState,OllamaAPI storage;
    class Guardrail,Reject security;
    class Tools,MemSearch,WebSearch,SysTools tool;
```
<p align="center"><u><sub>System Architecture Diagram</sub></u></p>

### Local Model Orchestration and Lifecycle Management
- **Automated Subprocess Control**: Harness automatically handles the startup and teardown of the local `llama-server`. It spawns a background subprocess utilizing GPU layer offloading (`-ngl 999`), a target context size (`--ctx-size 16384`), and binds to port `8000`.
- **Pre-flight Port Checking**: Prior to startup, the framework checks if port `8000` is active via a connection check to `http://localhost:8000/v1/models`. If a server is active, it skips launch to prevent port collision.
- **Teardown Cleanup**: Uses Python's `atexit` library to register a cleanup hook, ensuring the background `llama-server` process is terminated gracefully on terminal session exit.
- **Log Redirection**: The server output is redirected to `llama-server.log` in the project root to prevent stdout/stderr clutter in the main terminal interface.

### Input Guardrails
- **Prompt Injection Defense**: Every raw user input is processed by a dedicated `guard_node` utilizing `guard_llm` (configured for ShieldGemma) before entering the orchestrator state graph. It implements entropy/special-character ratio limits (ignoring safe math symbols) to catch obfuscation.
- **Inline Obfuscation Decoding**: The guardrail automatically intercepts short Base64 and Hex strings, decoding them inline to check against blacklisted payloads (e.g., `system prompt` or `bypass` directives) without relying purely on regex length limits.
- **Deterministic Routing**: If the guard model or heuristic checks classify the input as an injection attack, the graph aborts immediately and returns a rejection message, bypassing the main orchestration engine entirely.


```mermaid
graph TD
    %% Full Memory Array
    OlderChunks[Older Memory Chunks] -->|Query| TopK[Top-k Retrieval]
    RecentChunks[Recent 96k Memory]
    
    %% Assembling the Context Window
    TopK -->|Retrieved Context| ContextWindow[Context-Window]
    RecentChunks -->|Prompt + Active Context| ContextWindow
    
    %% Core Processing
    ContextWindow --> LLM[Large Language Model]
    LLM --> Output[Generate Response]
    
    %% Colourful Styling
    classDef memory fill:#ff9f43,stroke:#e67e22,stroke-width:3px,color:#fff,rx:10,ry:10;
    classDef retrieval fill:#1abc9c,stroke:#16a085,stroke-width:3px,color:#fff,rx:10,ry:10;
    classDef context fill:#9b59b6,stroke:#8e44ad,stroke-width:3px,color:#fff,rx:10,ry:10;
    classDef llmNode fill:#2980b9,stroke:#2c3e50,stroke-width:4px,color:#fff,padding:40px,rx:15,ry:15;
    classDef output fill:#27ae60,stroke:#2ecc71,stroke-width:3px,color:#fff,rx:10,ry:10;
    
    class OlderChunks,RecentChunks memory;
    class TopK retrieval;
    class ContextWindow context;
    class LLM llmNode;
    class Output output;
```
<p align="center"><u><sub>Retrieval & Sliding-Window based Context Management</sub></u></p>

### LLM-Driven Vector Memory (Qdrant)
- **Vector Searchable History**: The agent automatically backs up all conversational context to a persistent local Qdrant database (`qdrant_db`).
- **Storage Optimization**: Rather than storing duplicated text inside Qdrant vector chunks, Harness implements a lightweight JSON payload system storing pointers (`trace_id`, `prompt_id`, `response_id`). During retrieval, the engine intercepts these pointers and dynamically re-stitches the exact text directly from local audit logs.
- **LLM-Driven Retrieval Tool**: The orchestrator is equipped with a `search_memory` tool. Instead of an automated retrieval step, the LLM analyzes user queries and decides autonomously whether to search its past memories, prioritizing this action over internet searches.

### Invisible Search Delegation
- **Native Ollama Search**: Web searches are handled via a custom reasoning loop utilizing Ollama's native `web_search` and `web_fetch` tools, directly leveraging the `ollama` Python client. 
- **Port Usage**: The main orchestrator (`llama-server`) runs on `localhost:8000`, while the native search loop connects to the default Ollama API on `localhost:11434`.
- **Encapsulated State**: Intermediate tool execution sequences and search iterations are hidden from the user, returning only the final synthesized query results to the main orchestrator without cluttering the main message history.

### Context Compression and State Management
- **Token Tracking & Sliding Window**: The `Archival Node` strictly manages the active conversation history inside the LangGraph State. By tracking prompt, response, and total token usage, it enforces a sliding token window. If the conversation breaches the configured token limit, it sequentially prunes the oldest turns, relying on the Qdrant database to store and retrieve them seamlessly later.

### Gated Tool Execution
- **Interactive Permissions**: Tools containing write access or shell execution logic (`write_file` and `run_cmd`) are gated with terminal prompts. Execution halts programmatically and prompts for explicit `(Y/n)` permission. 
- **Workspace Default Directory**: The `write_file` tool operates with a default directory rule. Unless the user explicitly requests changes in the current directory, the system prepends `workspace/` and handles directory creation (`os.makedirs`) dynamically.
- **OOM Prevention in Shell Tools**: Heavy command tools like `search_code` leverage `subprocess.Popen` streams to chunk standard output, automatically terminating runaway recursive greps and limiting returned strings to ~100KB to protect the LLM context window and prevent Python memory crashes.


### Audit Tracing
- **Structured Log Files**: Every terminal session creates an immutable, timestamped audit log inside the `execution_traces/` directory.
- **Serialized Event Types**: Events are recorded in strict chronological JSON-Lines format containing `timestamp`, `event_type` (`reasoning`, `tool-call`, `tool-output`, `prompt`, `error`, `final-response`), and `content`. 

### CLI Visuals
- **ANSI Animation**: On startup, a beautifully animated ASCII rendition of the Harness logo is dynamically rendered to the terminal.
- **Tool Spinners**: Interactive, animated spinners notify the user during background operations like `Digging through memory...` and `Searching the web...`.

---

## Evaluations

Harness includes an automated evaluation suite to benchmark the orchestrator model against three golden-path datasets (`gp_tools`, `gp_hitl`, `gp_redteam`). The evaluation traces are independently graded by LLM judges `gemma4:31b-cloud` and `gpt-oss:120b-cloud`.

```mermaid
graph TD
    %% Datasets & Benchmarks
    subgraph Benchmarks ["Golden Path Benchmarks"]
        GPTools["gp_tools.json<br/>(Tool Use)"]
        GPHITL["gp_hitl.json<br/>(Human-in-the-Loop)"]
        GPRedteam["gp_redteam.json<br/>(Red-Teaming)"]
    end

    %% Evaluation Runner
    Benchmarks --> EvalHarness["Evaluation Runner<br/>(evals/run_evals.py)"]
    
    %% Simulation & Mock Inputs
    EvalHarness -- "gp_hitl: Stream Context" --> Simulator["LLM User Simulator<br/>(gemma4:31b-cloud)"]
    Simulator -- "Dynamic Responses / Feedback" --> EvalHarness
    EvalHarness -- "gp_tools / gp_redteam" --> MockInputs["Static Mock Inputs"]
    
    %% Sandbox & Execution
    EvalHarness --> Sandbox[("Isolated Sandbox<br/>evals/eval_env/")]
    EvalHarness -- "Streamed Input Events" --> HarnessGraph["Harness Agent Graph<br/>(src/graph.py)"]
    HarnessGraph -. "File Ops Containment" .-> Sandbox
    
    %% Evaluation & Judging
    HarnessGraph -- "Formatted Traces" --> JudgeRouter{"LLM Judges"}
    JudgeRouter --> JudgeGemma["gemma4:31b-cloud"]
    JudgeRouter --> JudgeGPT["gpt-oss:120b-cloud"]
    
    %% Output & Reporting
    JudgeGemma -- "PASS / FAIL" --> ResultsCSV[("evals/results.csv")]
    JudgeGPT -- "PASS / FAIL" --> ResultsCSV
    ResultsCSV --> SpiderGen["Visualizer<br/>(evals/spider.py)"]
    SpiderGen --> SpiderPlot["spider_plot.png"]

    %% Styling
    classDef primary fill:#4A90E2,stroke:#333,stroke-width:2px,color:#fff;
    classDef storage fill:#50E3C2,stroke:#333,stroke-width:2px,color:#000;
    classDef tool fill:#F5A623,stroke:#333,stroke-width:2px,color:#fff;
    classDef security fill:#E94B3C,stroke:#333,stroke-width:2px,color:#fff;
    
    class EvalHarness,HarnessGraph,SpiderGen primary;
    class Sandbox,ResultsCSV,SpiderPlot storage;
    class Simulator,JudgeGemma,JudgeGPT tool;
    class JudgeRouter,MockInputs security;
```
<p align="center"><u><sub>Automated Evaluation & Simulation Architecture</sub></u></p>

<p align="center"><img src="evals/spider_plot.png" width="600"></p>

**=== Evaluation Results Summary ===**
- **Tool Use**            : Gemma = 29/30 (96.7%), GPT = 29/30 (96.7%)
- **Red-Teaming**         : Gemma = 26/30 (86.7%), GPT = 26/30 (86.7%)
- **Human-in-the-Loop**   : Gemma = 27/30 (90.0%), GPT = 27/30 (90.0%)

### LLM User Simulator
To rigorously test the agent's Human-in-the-Loop (`gp_hitl`) capabilities, the evaluation harness employs a real-time **LLM User Simulator**. Instead of static string responses, a secondary `gemma4:31b-cloud` model dynamically plays the role of the user, generating approvals, denials, conversational tool feedback, and multi-turn context checks based on the dataset's objectives.

---

## Installation

1. **Prerequisites**: 
   - **Python 3.12+**.
   - **llama.cpp (`llama-server`)**: The orchestrator node runs the LLM via an OpenAI-compatible endpoint. This project uses Gemma-4 E4B served locally by `llama-server`, so the `llama-server` command must be installed and available in your `PATH`.
   - **Ollama**: Running locally (used for guardrails and web searches).
2. **Environment Setup**: Set up your Python virtual environment in the `.venv` directory and run dependencies setup.
3. **Configuration**: Edit `config.yaml` to point to the correct model paths:
   ```yaml
   default_orch_model: "<model-path>"
   default_search_model: "<model-name>"
   default_guard_model: "<model-name>"
   ```
4. **Command CLI Wrapper**: Install the CLI utility by running the setup script from the root directory:
   ```bash
   ./install.sh
   ```
   This generates a wrapper script at `~/.local/bin/harness` that handles path resolution, virtual environment activation, and configuration loading.

---

## Usage

Run the global CLI command from any terminal directory:
```bash
harness
```

- Submit queries at the `>` input prompt.
- The CLI displays the orchestrator's reasoning trace in gray italics followed by the final output.
- When background tasks run, the terminal displays animated text (e.g. `Searching the web...`).
- Type `/bye` to exit the CLI session and terminate background processes.

---

## Repository Layout

- **`src/graph.py`**: Defines the state machine graph, node functions, edge routing, and automated background model processes.
- **`src/tools.py`**: Implementation of permissions, workspace defaults, and the React sub-agent definitions.
- **`src/llm.py`**: LLM instances initialization and setup parameters.
- **`src/prompts.py`**: Defines system prompts and instructions (e.g., `SYSTEM_PROMPT`) for guiding the orchestrator LLM.
- **`src/utils.py`**: Utility functions for managing the `llama-server` subprocess lifecycle and session audit logging.
- **`src/audit.py`**: Contains the `AuditLogger` singleton handling the write logic for structured JSON logs.
- **`src/memory.py`**: Handles Qdrant initialization, async embeddings generation, chunking logic, and dynamic reconstruction from trace logs.
- **`src/cli.py`**: Contains the terminal UI framework, rendering engines, and animated spinners.
