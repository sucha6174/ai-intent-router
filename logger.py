import json
import os

log_file = os.path.join(
    os.path.dirname(__file__),
    "route_log.jsonl"
)

def log_route(intent, confidence, user_message, final_response):
    log_entry = {
        "intent": intent,
        "confidence": confidence,
        "user_message": user_message,
        "final_response": final_response
    }

    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")