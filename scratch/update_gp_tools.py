import json

tasks = []

# 1. Math and Logic Tasks
math_prompts = [
    ("Calculate the area of a circle with radius 10 (use 3.14 for pi) and write to math_0.txt", ["evaluate_math", "write_file"], ["math_0.txt"], "Agent must calculate 314.0"),
    ("Multiply 999999999 by 888888888 using long_int_multiply and write to math_1.txt", ["long_int_multiply", "write_file"], ["math_1.txt"], "Agent must calculate 888888887111111112"),
    ("Evaluate the expression '(50 * 12) + (2**10)' and write to math_2.txt", ["evaluate_math", "write_file"], ["math_2.txt"], "Agent must calculate 1624"),
    ("Calculate the modulus of 123456 by 789 and write to math_3.txt", ["evaluate_math", "write_file"], ["math_3.txt"], "Agent must calculate 363"),
    ("Multiply 12345 by 67890 using long_int_multiply and write to math_4.txt", ["long_int_multiply", "write_file"], ["math_4.txt"], "Agent must calculate 838102050"),
    ("Calculate 15 percent of 850 and write to math_5.txt", ["evaluate_math", "write_file"], ["math_5.txt"], "Agent must calculate 127.5"),
    ("Evaluate 5 factorial (5 * 4 * 3 * 2 * 1) and write to math_6.txt", ["evaluate_math", "write_file"], ["math_6.txt"], "Agent must calculate 120"),
    ("Calculate the square root of 1024 by evaluating '1024 ** 0.5' and write to math_7.txt", ["evaluate_math", "write_file"], ["math_7.txt"], "Agent must calculate 32.0"),
    ("Evaluate '100 / 3' and write to math_8.txt", ["evaluate_math", "write_file"], ["math_8.txt"], "Agent must calculate approximately 33.33"),
    ("Multiply 111111 by 222222 using long_int_multiply and write to math_9.txt", ["long_int_multiply", "write_file"], ["math_9.txt"], "Agent must calculate 24691308642")
]

for i, (prompt, expected_tool_calls, expected_files, eval_criteria) in enumerate(math_prompts):
    tasks.append({
        "task_id": f"tools_math_{i}",
        "description": f"Diverse math calculation {i}",
        "prompt": prompt,
        "expected_tool_calls": expected_tool_calls,
        "expected_files": expected_files,
        "evaluation_criteria": eval_criteria
    })

# 2. File and Code Reading Tasks
file_prompts = [
    ("List the contents of the 'src' directory and write to search_results_0.txt", ["list_dir", "write_file"], ["search_results_0.txt"], "Agent must use list_dir on src"),
    ("Search the codebase for 'def search_memory' and write the match to search_results_1.txt", ["search_code", "write_file"], ["search_results_1.txt"], "Agent must use search_code"),
    ("Read the file 'config.yaml' and write its contents to search_results_2.txt", ["read_file", "write_file"], ["search_results_2.txt"], "Agent must use read_file on config.yaml"),
    ("Search the codebase for 'OLLAMA_API_KEY' and write the matches to search_results_3.txt", ["search_code", "write_file"], ["search_results_3.txt"], "Agent must use search_code"),
    ("List the contents of the current directory '.' and write to search_results_4.txt", ["list_dir", "write_file"], ["search_results_4.txt"], "Agent must use list_dir"),
    ("Search the codebase for 'import yaml' and write the matched lines to search_results_5.txt", ["search_code", "write_file"], ["search_results_5.txt"], "Agent must use search_code"),
    ("Read the file 'pyproject.toml' and write it to search_results_6.txt", ["read_file", "write_file"], ["search_results_6.txt"], "Agent must use read_file"),
    ("Search the codebase for 'from .llm import' and write the files to search_results_7.txt", ["search_code", "write_file"], ["search_results_7.txt"], "Agent must use search_code"),
    ("List the contents of 'evals' directory and write to search_results_8.txt", ["list_dir", "write_file"], ["search_results_8.txt"], "Agent must use list_dir on evals"),
    ("Search the codebase for 'subprocess.run' and write the results to search_results_9.txt", ["search_code", "write_file"], ["search_results_9.txt"], "Agent must use search_code")
]

for i, (prompt, expected_tool_calls, expected_files, eval_criteria) in enumerate(file_prompts):
    tasks.append({
        "task_id": f"tools_file_read_{i}",
        "description": f"Diverse file operation {i}",
        "prompt": prompt,
        "expected_tool_calls": expected_tool_calls,
        "expected_files": expected_files,
        "evaluation_criteria": eval_criteria
    })

# 3. Tool Chaining and Advanced Operations
chain_prompts = [
    ("Run a web search for 'Python 3.12 release date' and write the result to git_status_0.txt", ["perform_web_search", "write_file"], ["git_status_0.txt"], "Agent must use web search"),
    ("Run git diff and write the output to git_status_1.txt", ["git_diff", "write_file"], ["git_status_1.txt"], "Agent must use git_diff"),
    ("Fetch the URL 'http://example.com' and write the excerpt to git_status_2.txt", ["fetch_url", "write_file"], ["git_status_2.txt"], "Agent must use fetch_url"),
    ("Search memory for 'database' and write the history to git_status_3.txt", ["search_memory", "write_file"], ["git_status_3.txt"], "Agent must use search_memory"),
    ("Use run_cmd to execute 'echo Hello World' and write the output to git_status_4.txt", ["run_cmd", "write_file"], ["git_status_4.txt"], "Agent must use run_cmd"),
    ("Check the git status and write it to git_status_5.txt", ["git_status", "write_file"], ["git_status_5.txt"], "Agent must use git_status"),
    ("Run a web search for 'Latest Ollama features' and write to git_status_6.txt", ["perform_web_search", "write_file"], ["git_status_6.txt"], "Agent must use web search"),
    ("Use run_cmd to execute 'ls -la' and write the output to git_status_7.txt", ["run_cmd", "write_file"], ["git_status_7.txt"], "Agent must use run_cmd"),
    ("Fetch the URL 'https://example.org' and write to git_status_8.txt", ["fetch_url", "write_file"], ["git_status_8.txt"], "Agent must use fetch_url"),
    ("Run git status, then run git diff, and write both to git_status_9.txt", ["git_status", "git_diff", "write_file"], ["git_status_9.txt"], "Agent must use git status and diff")
]

for i, (prompt, expected_tool_calls, expected_files, eval_criteria) in enumerate(chain_prompts):
    tasks.append({
        "task_id": f"tools_chain_{i}",
        "description": f"Diverse tool chaining {i}",
        "prompt": prompt,
        "expected_tool_calls": expected_tool_calls,
        "expected_files": expected_files,
        "evaluation_criteria": eval_criteria
    })

data = {"tasks": tasks}

with open("evals/golden_paths_v1/gp_tools.json", "w") as f:
    json.dump(data, f, indent=4)
