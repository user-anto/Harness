import os
import datetime
import json

class AuditLogger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AuditLogger, cls).__new__(cls)
            cls._instance.log_file = None
        return cls._instance

    def initialize(self):
        if self.log_file is not None:
            return
            
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_dir = os.path.join(base_dir, "execution_traces")
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{timestamp}.log")
        
        with open(self.log_file, "a") as f:
            f.write(f"--- Audit Log Started: {timestamp} ---\n")

    def _log_event(self, event_type: str, content: str):
        if not self.log_file:
            self.initialize()
            
        timestamp = datetime.datetime.now().isoformat()
        log_entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "content": content
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def log_prompt(self, prompt: str):
        self._log_event("prompt", prompt)

    def log_reasoning(self, reasoning: str):
        self._log_event("reasoning", reasoning)
        
    def log_tool_call(self, name: str, args: dict):
        self._log_event("tool-call", f"Function: {name}, Args: {json.dumps(args)}")
        
    def log_tool_output(self, name: str, output: str):
        self._log_event("tool-output", f"Function: {name}, Output: {output}")
        
    def log_error(self, error: str):
        self._log_event("error", error)
        
    def log_final_response(self, response: str):
        self._log_event("final-response", response)

audit_logger = AuditLogger()
