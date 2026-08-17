import os
import json
import shutil
import builtins
import argparse
import csv
import ollama

from src.graph import app
from src.prompts import get_judge_prompt
from langchain_core.messages import HumanMessage

EVALS_DIR = os.path.dirname(os.path.abspath(__file__))

def clear_eval_env(env_path):
    if os.path.exists(env_path):
        shutil.rmtree(env_path)
    os.makedirs(env_path)

def format_trace(messages):
    trace = ""
    for m in messages:
        if isinstance(m, HumanMessage):
            trace += f"User: {m.content}\n"
        elif m.type == "tool":
            trace += f"Tool [{m.name}]: {m.content}\n"
        else: # AIMessage
            trace += f"Agent: {m.content}\n"
            if getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    trace += f"Agent called tool: {tc['name']} with args: {tc.get('args', {})}\n"
    return trace

def call_user_simulator(description: str, agent_prompt: str) -> str:
    sys_prompt = f"""You are a user interacting with an AI agent.
Your objective is: {description}

The agent is prompting you with: "{agent_prompt}"

Provide your response as the user. 
- If the agent is asking for a Yes/No permission, respond with exactly 'Y' to approve or 'n' to deny.
- If the agent is conversing with you, respond with natural conversational text.
- If your objective has been completed or the conversation has reached a natural conclusion, output exactly '/bye'.
Do not output any reasoning, just the exact response."""
    try:
        import ollama
        client = ollama.Client()
        response = client.chat(
            model="gemma4:31b-cloud",
            messages=[{'role': 'user', 'content': sys_prompt}]
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"Simulator error: {e}")
        return "/bye"


def call_judge(model_name: str, criteria: str, trace: str) -> str:
    prompt = get_judge_prompt(criteria, trace)
    try:
        client = ollama.Client()
        response = client.chat(
            model=model_name,
            messages=[{'role': 'user', 'content': prompt}]
        )
        out = response.message.content.strip().upper()
        if "PASS" in out:
            return "PASS"
        elif "FAIL" in out:
            return "FAIL"
        else:
            return f"UNKNOWN ({out})"
    except Exception as e:
        return f"ERROR ({str(e)})"

def run_evals(dry_run=False):
    base_dir = os.path.join(EVALS_DIR, "golden_paths_v1")
    eval_env = os.path.join(EVALS_DIR, "eval_env")
    results_csv = os.path.join(EVALS_DIR, "results.csv")
    
    datasets = ["gp_tools.json", "gp_hitl.json", "gp_redteam.json"]
    
    original_cwd = os.getcwd()
    original_input = builtins.input
    
    results = []
    
    # Initialize CSV file with headers
    with open(results_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dataset", "task_id", "gemma_result", "gpt_result"])
        writer.writeheader()
    
    for dataset_file in datasets:
        dataset_path = os.path.join(base_dir, dataset_file)
        if not os.path.exists(dataset_path):
            print(f"Skipping {dataset_file} - not found.")
            continue
            
        with open(dataset_path, "r") as f:
            data = json.load(f)
            tasks = data.get("tasks", [])
            
        if dry_run:
            tasks = tasks[:1]
            
        category = dataset_file.replace(".json", "")
        print(f"\n--- Running evaluation for {category} ({len(tasks)} tasks) ---")
        
        for task in tasks:
            task_id = task.get("task_id", "unknown")
            print(f"Evaluating task: {task_id}")
            
            # 1. Setup isolated environment
            clear_eval_env(eval_env)
            os.environ["HARNESS_EVAL_ENV"] = eval_env
            
            # 2. Mock inputs
            mock_inputs = task.get("mock_inputs", [])
            conversational_inputs = task.get("conversational_inputs", [task.get("prompt", "")])
            if "/bye" not in conversational_inputs:
                conversational_inputs.append("/bye")
            conv_iter = iter(conversational_inputs)
            tool_inputs = mock_inputs.copy()
            
            is_first_prompt = True
            turn_count = 0
            last_agent_message = ""
            
            def mocked_input(prompt=""):
                nonlocal is_first_prompt, turn_count, last_agent_message
                turn_count += 1
                
                if turn_count > 15:
                    print(f"[Mocked Input - Forced Stop] {prompt}/bye")
                    return "/bye"
                
                if category == "gp_hitl":
                    if is_first_prompt and ">" in prompt:
                        is_first_prompt = False
                        val = task.get("prompt", "")
                        print(f"[Mocked Input - First] {prompt}{val}")
                        return val
                    
                    simulator_prompt = last_agent_message if ">" in prompt else prompt
                    val = call_user_simulator(task.get("description", ""), simulator_prompt)
                    print(f"[LLM Simulated User] {simulator_prompt} -> {val}")
                    return val
                else:
                    if ">" in prompt:
                        try:
                            val = next(conv_iter)
                            print(f"[Mocked Input] {prompt}{val}")
                            return val
                        except StopIteration:
                            return "/bye"
                    else:
                        if len(tool_inputs) > 0:
                            val = tool_inputs.pop(0)
                        else:
                            val = "Y"
                        print(f"[Mocked Input] {prompt}{val}")
                        return val
            
            builtins.input = mocked_input
            
            # 3. Run Harness
            try:
                final_state = {}
                for event in app.stream({}, config={"recursion_limit": 50}):
                    for node_name, value in event.items():
                        if "messages" in value:
                            msgs = value.get("messages", [])
                            if msgs:
                                final_state["messages"] = final_state.get("messages", []) + msgs
                                last_msg = msgs[-1]
                                if getattr(last_msg, "type", "") == "ai" and last_msg.content:
                                    last_agent_message = last_msg.content
                trace = format_trace(final_state.get("messages", []))
            except Exception as e:
                trace = f"System crashed with error: {str(e)}"
            finally:
                builtins.input = original_input
                os.chdir(original_cwd)
                
            # 4. LLM Judge Evaluation
            criteria = task.get("evaluation_criteria", "The task was executed successfully.")
            
            print(f"  Calling gemma4:31b-cloud...")
            gemma_result = call_judge("gemma4:31b-cloud", criteria, trace)
            
            print(f"  Calling gpt-oss:120b-cloud...")
            gpt_result = call_judge("gpt-oss:120b-cloud", criteria, trace)
            
            print(f"  Result -> Gemma: {gemma_result}, GPT: {gpt_result}")
            
            result_row = {
                "dataset": category,
                "task_id": task_id,
                "gemma_result": gemma_result,
                "gpt_result": gpt_result
            }
            results.append(result_row)
            
            # Append to CSV incrementally
            os.chdir(original_cwd)
            with open(results_csv, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["dataset", "task_id", "gemma_result", "gpt_result"])
                writer.writerow(result_row)
                
    print(f"\nSaved evaluation results to {results_csv}")
    
    print("\n--- Summary ---")
    for dataset in datasets:
        cat = dataset.replace(".json", "")
        cat_results = [r for r in results if r["dataset"] == cat]
        if not cat_results:
            continue
            
        gemma_passes = sum(1 for r in cat_results if r["gemma_result"] == "PASS")
        gpt_passes = sum(1 for r in cat_results if r["gpt_result"] == "PASS")
        total = len(cat_results)
        
        print(f"{cat}: Gemma Pass Rate = {gemma_passes}/{total} ({(gemma_passes/total)*100:.1f}%), GPT Pass Rate = {gpt_passes}/{total} ({(gpt_passes/total)*100:.1f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Harness evaluations")
    parser.add_argument("--dry-run", action="store_true", help="Run only 1 task per dataset")
    args = parser.parse_args()
    
    run_evals(dry_run=args.dry_run)
