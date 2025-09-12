from datetime import datetime, timezone

def now_iso():
    return datetime.now(timezone.utc).isoformat()

def mask_msisdn(msisdn: str) -> str:
    if len(msisdn) >= 7:
        return msisdn[:3] + "***" + msisdn[-4:]
    return msisdn

def parse_name_list(names_str: str) -> list:
    if not names_str.strip():
        return []
    return [name.strip() for name in names_str.split(",") if name.strip()]


import json
import time
from pathlib import Path

def append_jsonl(path, data):
    data["timestamp"] = now_iso()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def read_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def debounce_message(msg_id, seen_set, seen_path):
    if msg_id in seen_set:
        return True
    seen_set.add(msg_id)
    with open(seen_path, "w", encoding="utf-8") as f:
        json.dump(list(seen_set), f, ensure_ascii=False)
    return False

def get_employee_name_by_msisdn(msisdn, employees):
    """Return employee name for a given msisdn, or None if not found."""
    for name, info in employees.items():
        if info.get("msisdn") == msisdn:
            return name
    return None
