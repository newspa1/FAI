import json
import os

def log_state_action(state, action, file_path="training_data.jsonl"):
    with open(file_path, "a") as f:
        f.write(json.dumps({"state": state, "action": action}) + "\n")